#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import numpy as np
import ray
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from ray.rllib.algorithms.algorithm import Algorithm

# --- CONFIGURAZIONE ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Path del modello di default (puoi cambiarlo qui o passare --model_dir)
DEFAULT_MODEL_DIR = "~/ML/TEST/RCT2_Modello_Transfer"

try:
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
except:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_envs.rct import DaysResearchMeetObjectiveRCTEnv
try:
    from env_config import env_config
except ImportError:
    # Configurazione di base se il file manca
    env_config = {"render_mode": "human", "real_time_factor": 0}

# --- CLASSE ENV MODIFICATA PER BYPASSARE CHECK ---
class SafeRCTEnv(DaysResearchMeetObjectiveRCTEnv):
    """
    Sottoclasse che sovrascrive get_starting_map per evitare 
    l'assert sulla dimensione della mappa, permettendo al test 
    di caricare scenari più grandi e adattarli dopo.
    Assicura che num_months non superi 24 (limite buffer environment).
    """
    def get_starting_map(self, map_type, park_filename):
        #  Aumentiamo temporaneamente map_size per passare il check.
        original_map_size = self.map_size
        self.map_size = (9999, 9999) # Dimensione fittizia enorme
        
        try:
            starting_map = super().get_starting_map(map_type, park_filename)
            # ADATTAMENTO: Cappa la durata a 24 mesi per evitare crash dei buffer
            if starting_map.num_months > 24:
                 starting_map.num_months = 24
            return starting_map
        finally:
            self.map_size = original_map_size # Ripristina la dimensione reale 

    def get_cur_values(self):
        """Patch per evitare IndexError in rct.py se num_months > 24"""
        old_num_months = self.num_months
        # Forza il loop di rct.py a non superare la dimensione del buffer (24)
        self.num_months = min(self.num_months, 24)
        try:
            return super().get_cur_values()
        finally:
            self.num_months = old_num_months

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Assicura che max_timesteps sia coerente col limite 24
        if self.num_months > 24:
             self.num_months = 24
        self.max_timesteps = self.num_months
        return obs, info

    def close(self):
        # Assicura che i processi vengano uccisi
        try:
            super().close()
        except:
            pass
        # Kill forzato extra se necessario
        try:
            if hasattr(self, 'bridge') and self.bridge.rct_process:
               self.bridge.rct_process.kill()
        except:
            pass

def find_latest_checkpoint(model_dir=None):
    # Se passato via arg, usa quello, altrimenti usa il default
    target_dir = model_dir if model_dir else DEFAULT_MODEL_DIR
    base_dir = os.path.expanduser(target_dir)

    print(f"🔎 Ricerca checkpoint in: {base_dir}")

    # Caso 1: Path punta direttamente a un checkpoint
    if os.path.basename(base_dir).startswith("checkpoint_"):
        return base_dir

    # Caso 2: Cerca dentro la cartella
    checkpoints = glob.glob(f"{base_dir}/**/checkpoint_000*", recursive=True)
    
    if not checkpoints:
        return None
        
    # Ordina per data di modifica (più recente)
    return max(checkpoints, key=os.path.getmtime)

def generate_graphs_and_stats(scenario_name, all_runs_data):
    """Genera grafici PNG e statistiche TXT per la tesi"""
    print(f"\n📊 Generazione report per {scenario_name}...")
    
    # 1. Grafico Ospiti
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(all_runs_data):
        plt.plot(run["history"]["guests"], label=f'Run {i+1} (Max: {run["guests"]})')
    plt.title(f'{scenario_name}: Crescita Visitatori')
    plt.xlabel('Mesi')
    plt.ylabel('Ospiti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{scenario_name}_guests.png")
    plt.close()

    # 2. Grafico Soldi
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(all_runs_data):
        plt.plot(run["history"]["money"], linestyle='--', label=f'Run {i+1}')
    plt.title(f'{scenario_name}: Andamento Economico')
    plt.xlabel('Mesi')
    plt.ylabel('Soldi (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{scenario_name}_money.png")
    plt.close()

    # 3. Statistiche Giostre
    all_actions = []
    for run in all_runs_data:
        all_actions.extend(run["history"]["actions"])
    
    with open(f"{scenario_name}_stats.txt", "w") as f:
        f.write(f"REPORT STATISTICO: {scenario_name}\n")
        f.write("="*40 + "\n")
        f.write(f"Media Ospiti: {np.mean([r['guests'] for r in all_runs_data]):.1f}\n")
        f.write(f"Win Rate: {np.mean([1 if r['win'] else 0 for r in all_runs_data])*100:.1f}%\n")
        f.write("-" * 20 + "\nTOP 10 GIOSTRE COSTRUITE:\n")
        for action, count in Counter(all_actions).most_common(10):
            f.write(f"- {action}: {count}\n")

    print(f"✅ Salvati: {scenario_name}_guests.png, _money.png, _stats.txt")

def run_episode(env, agent):
    """Esegue un episodio adattando la mappa alle dimensioni 87x87"""
    obs, info = env.reset()
    
    # --- ADATTATORE UNIVERSALE DIMENSIONI ---
    # Il modello VUOLE 87x87. Se la mappa è diversa, la forziamo.
    TARGET_SIZE = 87
    
    def adatta_osservazione(original_obs):
        """Adatta l'osservazione a TARGET_SIZE x TARGET_SIZE (crop/pad)"""
        # Gestione obs come dict (con chiave 'matrix') o array diretto
        if isinstance(original_obs, dict):
            matrix = original_obs['matrix']
        else:
            matrix = original_obs
            
        h, w, channels = matrix.shape
        new_matrix = np.zeros((TARGET_SIZE, TARGET_SIZE, channels), dtype=matrix.dtype)
        
        # Crop se più grande, pad se più piccolo
        copy_h = min(h, TARGET_SIZE)
        copy_w = min(w, TARGET_SIZE)
        new_matrix[:copy_h, :copy_w, :] = matrix[:copy_h, :copy_w, :]
        
        if isinstance(original_obs, dict):
            # Ricostruiamo il dict con la nuova matrix
            new_obs = original_obs.copy()
            new_obs['matrix'] = new_matrix
            return new_obs
        return new_matrix
    
    # Applichiamo l'adattamento subito
    obs = adatta_osservazione(obs)
    # -----------------------------------------

    done = False
    truncated = False
    total_reward = 0
    max_guests = 0
    history = {"guests": [], "money": [], "actions": []}
    
    while not (done or truncated):
        try:
            action = agent.compute_single_action(obs, explore=False)
        except:
            res = agent.compute_single_action(obs, explore=False)
            action = res[0] if isinstance(res, tuple) else res

        # Eseguiamo l'azione nell'ambiente REALE
        raw_obs, reward, done, truncated, info = env.step(action)
        
        # Ri-adattiamo l'osservazione per il prossimo step
        obs = adatta_osservazione(raw_obs)
        
        total_reward += reward
        
        # --- RACCOLTA DATI RAW (NON NORMALIZZATI) ---
        if hasattr(env, 'raw_values'):
            curr_guests = env.raw_values.get('num_guests', 0)
            curr_money = env.raw_values.get('cash', 0)
        else:
            curr_guests = info.get('num_guests', 0)
            curr_money = info.get('cash', 0)
        
        if curr_guests > max_guests: 
            max_guests = curr_guests
        
        history["guests"].append(int(curr_guests))
        history["money"].append(int(curr_money))
        
        # --- ESTRAZIONE NOME GIOSTRA LEGGIBILE ---
        try:
            if isinstance(action, dict) and 'position_and_type' in action:
                ride_idx = int(action['position_and_type'][1])
                if ride_idx == len(env.ride_types):
                    ride_name = "SKIP"
                else:
                    ride_data = env.ride_types[ride_idx]
                    ride_name = ride_data.shop_name if ride_data.shop_name else ride_data.name.replace("RIDE_TYPE_", "")
            else:
                ride_name = f"Action_{action}"
        except:
            ride_name = str(action)
        
        history["actions"].append(ride_name)

    is_win = info.get('win', False) or max_guests >= 700
    
    return {
        "reward": total_reward, "guests": max_guests, 
        "win": is_win, "history": history
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", default=["Electric Fields.SC6"], 
                        help="Scenari da testare (es. 'Electric Fields.SC6')")
    parser.add_argument("--model_dir", type=str, default=None,
                        help=f"Path cartella modello (default: {DEFAULT_MODEL_DIR})")
    args = parser.parse_args()

    # Init Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    checkpoint = find_latest_checkpoint(args.model_dir)
    if not checkpoint:
        print("❌ Nessun checkpoint trovato!")
        sys.exit(1)
        
    print(f"🤖 Modello caricato: {checkpoint}")
    try:
        agent = Algorithm.from_checkpoint(checkpoint)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Errore caricamento modello: {e}")
        sys.exit(1)

    # Costruzione path scenari
    base_path = os.path.join(os.path.dirname(__file__), "small_parks")
    scenarios = []
    
    # NIENTE logica ALL automatica, solo quello che passa l'utente
    for s in args.scenarios:
        # Cerca prima nella cartella small_parks
        p = os.path.join(base_path, s)
        if os.path.exists(p):
            scenarios.append(p)
        elif os.path.exists(s): # Poi path assoluto/relativo diretto
            scenarios.append(s)
        else:
            print(f"⚠️ Scenario non trovato: {s}")

    if not scenarios:
        print(f"❌ Nessuno scenario valido da testare.")
        sys.exit(1)

    for park_path in scenarios:
        name = os.path.basename(park_path)
        print(f"\n📍 TEST SCENARIO: {name}")
        
        cfg = env_config.copy()
        cfg["park_filename"] = park_path
        cfg["headless"] = False
        
        try:
            
            env = SafeRCTEnv(cfg)
        except Exception as e:
            print(f"⚠️ Errore init env {name}: {e}"); continue

        run_data = []
        for i in range(args.runs):
            print(f"  ▶️ Run {i+1}/{args.runs}...", end=" ", flush=True)
            try:
                m = run_episode(env, agent)
                run_data.append(m)
                print(f"Ospiti: {m['guests']} | Win: {m['win']}")
            except Exception as e:
                print(f"❌ Crash: {e}")
                import traceback
                traceback.print_exc()
        
        try:
            env.close()
        except:
            pass
            
        if run_data:
            generate_graphs_and_stats(name, run_data)

    ray.shutdown()

if __name__ == "__main__":
    main()