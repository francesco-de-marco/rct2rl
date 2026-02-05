# Reinforcement Learning su RollerCoaster Tycoon (OpenRCT2)

![Project Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Ray RLlib](https://img.shields.io/badge/Ray-RLlib-blueviolet)

Questo progetto presenta una replica sperimentale e un'analisi di un approccio di **Deep Reinforcement Learning** applicato al videogioco gestionale **OpenRCT2**. L'agente utilizza l'algoritmo **PPO (Proximal Policy Optimization)** con una rete neurale multi-modale per gestire parchi divertimento, ottimizzando il layout e la gestione economica per raggiungere obiettivi specifici di visitatori e rating.

## 🎯 Obiettivi

*   **Validazione dell'algoritmo PPO** in ambienti complessi e stocastici come RCT.
*   **Gestione spaziale a griglia**: L'agente "vede" il parco tramite mappe di calore (eccitazione, intensità, altezza) elaborate da una CNN Encoder-Decoder.
*   **Action Masking**: Integrazione di vincoli logici per impedire azioni non valide (es. costruire in acqua o fuori mappa).
*   **Replica su Hardware Consumer**: Adattamento degli iperparametri originali per il training su GPU singola (RTX 3050 Ti) con risorse limitate.

## 🧠 Architettura del Sistema

Il sistema disaccoppia l'agente dall'ambiente di gioco utilizzando **OpenRCT2** in modalità headless e comunicando tramite socket ZeroMQ.

*   `pathrl.py`: **Entry Point**. Inizializza Ray e avvia il loop di training PPO.
*   `gen_envs/rct.py`: Wrapper **Gymnasium** che traduce gli stati di gioco in tensori per la rete neurale.
*   `bridge.py`: Gestisce la comunicazione **ZeroMQ** con il processo OpenRCT2 C++.
*   `visionnet2d.py`: Rete **CNN Encoder-Decoder** che processa la mappa 87x87 del parco.
*   `rl_model.py`: Implementa la policy network e l'**Action Masking**.

## 🚀 Installazione

### Prerequisiti
*   Python 3.10
*   OpenRCT2 installato e configurato (inclusi i file asset originali di RCT2).
*   Linux (testato su Ubuntu).

### Setup
1. Clona il repository:
   ```bash
   git clone https://github.com/francesco-de-marco/rct2rl.git
   cd rct2rl
   ```

2. Crea un virtual environment e installa le dipendenze:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Verifica i percorsi in `paths.py`:
   Assicurati che `RCT_EXECUTABLE` punti al tuo binario di OpenRCT2.

## 💻 Utilizzo

Il training può essere avviato in diverse modalità a seconda della complessità dello scenario (vedi gerarchia ambienti in `rct.py`).

Comando base:
```bash
python pathrl.py <mode>
```

Dove `<mode>` indica il livello di difficoltà:
*   `1`: **RCTEnv** (Simulazione libera, training continuativo senza Game Over).
*   `2`: **MeetObjectiveRCTEnv** (Aggiunge condizioni di vittoria/sconfitta basate su target ospiti).
*   `3`: **ResearchMeetObjectiveRCTEnv** (Aggiunge la meccanica della ricerca dei ride).
*   `4`: **DaysResearchMeetObjectiveRCTEnv** (Simulazione realistica giorno-per-giorno).

## 📊 Risultati Sperimentali

Il training è stato ottimizzato rispetto alla configurazione originale applicando tre interventi principali per garantirne la fattibilità su hardware consumer:

1.  **Adattamento Iperparametri**: Riduzione del *training batch size* (da 512 a 256) e fissaggio dei worker a 2 per operare entro i limiti di VRAM della GPU (RTX 3050 Ti).
2.  **Stabilità delle Risorse**: Vincoli espliciti sulla memoria dell'object store di Ray per prevenire crash OOM (*Out Of Memory*) e saturazione dello swap.
3.  **Hybrid Reward Function**: Modifica della funzione di ricompensa per combinare il *shaping* denso originale con segnali sparsi terminali (bonus vittoria/sconfitta), accelerando la convergenza su orizzonti temporali ridotti.
4.  **CBAM Attention**: Integrazione del modulo *Convolutional Block Attention Module* (CBAM) nella rete visuale per migliorare l'estrazione delle feature spaziali critiche.
5.  **Transfer Learning**: Implementazione di pipeline di *Fine-Tuning* che hanno permesso di adattare il modello a nuovi scenari (es. *Crazy Castle*).

> **Nota sulla Replica**: Per i dettagli completi sulla codebase di base e sulla procedura di replica originale, fare riferimento al [README della ricerca originale](https://github.com/campbelljc/rctrl) citato nei crediti.

## 👤 Autore

**Francesco De Marco**
Progetto di Machine & Deep Learning - A.A. 2025/2026

---
*Credit: Basato sulla ricerca originale e sulla base di codice `https://github.com/campbelljc/rctrl`.*
