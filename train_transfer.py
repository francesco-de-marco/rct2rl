import random
import os, sys
random.seed(1337)
from collections import defaultdict

os.environ["TF_USE_LEGACY_KERAS"]="1"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
#os.environ["RAY_record_ref_creation_sites"] = "1"
#os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_local_fs_capacity_threshold"]="1"

import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity

import ray._private.utils
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF2Policy

from paths import *
if 'cluster' in sys.argv:
    ray.init(
        address="auto",
        runtime_env={
            "working_dir": BASE_DIR,
            'excludes': [f'{BASE_DIR}/default_policy/policy_state.pkl']
        },
    )

from train import *
from env_config import *
from rl_model import TargetGridActionMaskModel

# --- CONFIGURAZIONE TRANSFER LEARNING ---
# 1. Carichiamo lo scenario Dusty Greens
env_config["park_filename"] = "small_parks/Crazy Castle.SC6"

# 2. Carichiamo il "Cervello" del Modello Finale
# (Percorso relativo rispetto a BASE_DIR=/home/francesco/ML/rctrl/)
env_config["load_model"] = "../../Desktop/RCT2_Modello_Finale/checkpoint_000063"
print(f"--- TRANSFER LEARNING SETUP ---")
print(f"Scenario: {env_config['park_filename']}")
print(f"Loading Model from: {env_config['load_model']}")
# ----------------------------------------

class MyPPOTFPolicy(PPOTF2Policy):
    def loss(self, model, dist_class, train_batch):
        # Call the original PPO loss method (from PPOTFPolicy).
        base_loss = super().loss(model, dist_class, train_batch)
        
        # Now, if your model has an action_embed_model that implements custom_loss,
        # call it to add the auxiliary ranking loss.
        if hasattr(model, "action_embed_model") and hasattr(model.action_embed_model, "custom_loss"):
            total_loss = model.action_embed_model.custom_loss(base_loss, train_batch)
        else:
            total_loss = base_loss
        
        return total_loss

    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        stats.update(self.model.action_embed_model.metrics())
        return stats


MAX_STATS = 1000
@ray.remote(name='metrics')
class RLStats:
    def __init__(self):
        self.stats = {}
        self.configs = defaultdict(list)
        self.success_rates = defaultdict(list)
        self.kde = {}
        self.can_sample = False

        self.vals_by_park = {}
        self.goals_by_park = {}

        self.ct = 0

        self.park_stats = {}
    
    def print_sizes(self):
        total = 0
        for attr in ['stats', 'configs', 'success_rates', 'kde', 'vals_by_park', 'goals_by_park', 'park_stats']:
            obj = getattr(self, attr)
            size = sys.getsizeof(obj)
            print(f"Size of {attr}: {size} bytes")
            total += size
        print(f"Total: {total} bytes")

    def episode_ended(self, park_filename, target_name, target_val, actual_val):
        if park_filename not in self.park_stats:
            self.park_stats[park_filename] = defaultdict(list)
        self.park_stats[park_filename][target_name].append(actual_val)
        #self.print_sizes()

    def get_park_stats(self):
        return self.park_stats

    def add_stat(self, park_filename, target, goal, actual):
        park_filename = park_filename.split('/')[-1]
        if park_filename not in self.vals_by_park:
            self.vals_by_park[park_filename] = defaultdict(list)
            self.goals_by_park[park_filename] = defaultdict(list)
        self.vals_by_park[park_filename][target].append(actual)
        self.goals_by_park[park_filename][target].append(goal)

        self.configs[target].append(goal)
        self.success_rates[target].append(abs(goal - actual))
        if len(self.configs[target]) > MAX_STATS:
            self.configs[target] = self.configs[target][1:]
            self.success_rates[target] = self.success_rates[target][1:]
    
    def update_sampling(self):
        #if not env_config["sample_targets"]:
        return
        # Update KDE model
        for target in self.configs:
            self.kde[target] = KernelDensity(kernel='gaussian', bandwidth=0.1)
            data = np.array(self.configs[target]).reshape(-1, 1)
            weights = np.array([max(1 - rate[0], 1e-3) for rate in self.success_rates[target]]).reshape(-1)
            self.kde[target].fit(data, sample_weight=weights)
            self.can_sample = True
    
    def sample(self, target):
        if not env_config["sample_targets"]:
            assert False
        assert target in self.kde # is not None
        # Generate a large number of samples from KDE
        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        log_density = self.kde[target].score_samples(x)
        density = np.exp(log_density)
        
        # Inverse the density to give higher weight to lower success rates
        inverse_density = 1.0 / (density + 1e-3)  # add a small constant to avoid division by zero
        inverse_density /= inverse_density.sum()  # normalize to get a valid probability distribution
        
        # Sample from the inverse density
        sampled_indices = np.random.choice(np.arange(1000), size=1000, p=inverse_density)
        return x[sampled_indices, 0]
    
    def can_sample(self):
        return self.can_sample
    
    def update(self, d):
        self.stats.update(d)
    
    def get(self):
        return self.stats

metric_stats = RLStats.remote()

alg = 'PPO'
env = envs[sys.argv[1]]

temp_env = env(env_config)
_, action_space = temp_env.get_action_space()
obs_space = temp_env.get_observation_space()
temp_env.close()

def policy_mapping_fn(x, y, worker):
    return "default_policy"

config = (
    PPOConfig()
    .training(
        model={
            "custom_model": TargetGridActionMaskModel,
            "custom_model_config": {
                "size": env_config["map_size"],
                "env": env,
            }
        },
        #lambda_=0.99,
        clip_param=0.3,
        entropy_coeff=0.02,
        kl_coeff=0.2,
        kl_target=0.03,
        lr=5e-5,
        num_sgd_iter=20,
        vf_clip_param=100.0,
        vf_loss_coeff=1.0,
        sgd_minibatch_size=64, # minibatch_size
        train_batch_size=256,
    )
    .experimental(_enable_new_api_stack=False)
    .environment(env, env_config=env_config, disable_env_checking=True)
    #.env_runners(num_env_runners=NUM_ROLLOUT_WORKERS)
    .rollouts(
        num_rollout_workers=2,
        rollout_fragment_length=32,
        sample_timeout_s=600,
    )
    .callbacks(MyCallback)
    .framework("tf2")
    .resources(num_gpus=1)
)

if __name__ == '__main__':
    # kill all OpenRCT2 processes
    os.system("pkill -f OpenRCT2")

    # LIMITIAMO A 4GB o 5GB MAX. 
    # Questo assicura che Ray non provi mai a scrivere sull'USB.
    if not ray.is_initialized():
        ray.init(
            num_gpus=1,
            # Limite 5GB. Se lo supera, Ray rallenta ma non deve esplodere su disco.
            object_store_memory=5000000000,
            _system_config={
                "max_io_workers": 2,  # Riduciamo i thread che scrivono su disco
            }
        )
        
    stop = {
        "training_iteration": 1000,     # ← 60000 → 1000
        "timesteps_total": 500000,      # ← 1200000 → 200000
        #"episode_reward_mean": 8,       # ← 60000 → 8
    }
    
    tuner = tune.Tuner(
        alg,
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=None,
                checkpoint_at_end=True,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",                      
                checkpoint_frequency=5,
            )
        ),
        tune_config = tune.TuneConfig(reuse_actors=True)
    )
    results = tuner.fit()
