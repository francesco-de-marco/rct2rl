import sys
import random
import pickle

import ray
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from paths import *
from gen_envs.rct import *
from env_config import env_config

print("--- LO SCRIPT STA PARTENDO ---")

class MyCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        values = episode.last_info_for() #episode.last_raw_obs_for() #
        #values = episode.last_info_for()
        for name, val in values.items():
            if name in ['action_mask', 'matrix'] or 'action_mask' in name:
                continue
            episode.custom_metrics[name] = val

        stats = ray.get_actor("metrics")
        env = base_env.get_sub_environments()[0]
        park_filename = env.chosen_park.filename.split('/')[-1]

        if park_filename[0].isdigit():
            park_filename = park_filename.split('-')[1]
        
        for target in env.target_ranges:
            #episode.custom_metrics[f'{target.name}_error'] = abs(env.targets[target.name] - values[target.name])
            episode.custom_metrics[f'{target.name}_{park_filename}'] = values[target.name]
            #env.park_stats[park_filename][target.name].append(values[target.name])
            stats.episode_ended.remote(park_filename, target.name, env.targets[target.name], values[target.name])
            #train.report(f'{target}_error', abs(env.targets[target] - values[target]))
            stats.add_stat.remote(env.chosen_park.filename, target.name, env.targets[target.name], values[target.name])
    
    def on_train_result(self, *, result: dict, **kwargs):
        stats = ray.get_actor("metrics")
        stats.update.remote(result['custom_metrics'])
        stats.update_sampling.remote()

    def on_algorithm_init(self, *, algorithm, **kwargs):
        if "load_model" not in env_config:
            return
        policy = Policy.from_checkpoint(
            BASE_DIR + env_config["load_model"] #, policy_ids=["main"]
        )
        print(policy)

        algorithm.set_weights({'default_policy': policy['default_policy'].get_weights()})
        algorithm.workers.sync_weights()

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id,
                                  policies, postprocessed_batch, **kwargs):
        return
    
        # Option 1: Check the in-memory size using pickle serialization.
        serialized = pickle.dumps(postprocessed_batch)
        size_bytes = len(serialized)
        print(f"Postprocessed batch size (serialized): {size_bytes / (1024 * 1024):.2f} MB")
        
        # Option 2: Examine individual components if your batch is a dict.
        if isinstance(postprocessed_batch, dict):
            for key, value in postprocessed_batch.items():
                # For numpy arrays, you can inspect the shape and data type.
                if isinstance(value, np.ndarray):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}, nbytes={value.nbytes}")
                else:
                    print(f"{key}: type={type(value)}, size approx={sys.getsizeof(value)} bytes")

envs = {
    '1': RCTEnv,
    '2': MeetObjectiveRCTEnv,
    '3': ResearchMeetObjectiveRCTEnv,
    '4': DaysResearchMeetObjectiveRCTEnv,
}
