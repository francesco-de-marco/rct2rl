from collections import namedtuple, OrderedDict

import numpy as np
import gymnasium as gym
from tabulate import tabulate
from gymnasium.spaces import Box, Dict
from ray.rllib.utils.spaces.repeated import Repeated

from gen_envs.consts import *

MAX_VALUE = 1

Target = namedtuple("Target", "name min_val max_val active weight")

class BasePathEnv(gym.Env):
    target_ranges = []
    active_target_ranges = [target for target in target_ranges if target.active]
    num_matrix_channels = 1
    num_flat_channels = len(active_target_ranges)*2

    def __init__(self, config):
        self.map_size = config.get('map_size')
        assert self.map_size[0] % 3 == 0 and self.map_size[1] % 3 == 0
        self.ride_range = config.get('ride_range')
        self.ride_size = config.get('ride_size')
        self.stop_early = config.get('stop_early', False)
        self.verbose = config.get('verbose', False)
        self.action_type = config.get('action_type')
        assert self.action_type in ['wide', 'turtle', 'wide_turtle'], self.action_type
        self.novelty_type = config.get('novelty_type', 'all')
        assert self.novelty_type in ['corner', 'all'], self.novelty_type
        self.novelty_point_type = config.get('novelty_point_type', 'all')
        assert self.novelty_point_type in ['all', 'rides']
        #self.target_weights = config.get('target_weights', {})
        self.ignore_paths_for_novelty = config.get('ignore_paths_for_novelty', False)
        
        self.max_timesteps = config.get('max_timesteps', 5)
        
        self.weight_averages = config.get('weight_averages', False)
        self.sample_targets = config.get('sample_targets', False)
        self.learn = True
        
        #if hasattr(config, 'worker_index'):
        #    random.seed(config.worker_index)
        
        self.sample_vals = {}
        
        self.obs_size = self.map_size if self.action_type == 'wide' else (self.map_size[0]*2-1, self.map_size[1]*2-1)
        
        self.num_actions, self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
    
    def get_action_space(self):
        if self.action_type in ['wide', 'wide_turtle']:
            num_actions = self.map_size[0] * self.map_size[1] #* 2
        elif self.action_type == 'turtle':
            num_actions = 4 #* 2
        
        return num_actions, Discrete(self.num_actions)
    
    def get_observation_space(self):
        return Dict({
            "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=bool),
            "matrix": Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(*self.obs_size, self.num_matrix_channels), dtype=np.float32),
            "flat": Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(self.num_flat_channels,), dtype=np.float32),
        
            # metrics for tensorboard.
            **{f'{target.name}': Box(-1, 10000, shape=(1,), dtype=np.float32) for target in self.target_ranges},
        })    

    def in_range(self, node, delta):
        return 0 <= node[0] + delta[0] < self.map_size[0] and 0 <= node[1] + delta[1] < self.map_size[1]
    
    def _fix_action_mask(self):
        raise NotImplementedError
    
    def _get_matrix(self):
        if self.action_type == 'wide':
            return self.matrix
        elif self.action_type in ['turtle', 'wide_turtle']:
            def center_array(original_array, i, j):
                n, m, z = original_array.shape
                new_shape = (2*n - 1, 2*m - 1, z)
                new_array = np.zeros(new_shape, dtype=original_array.dtype)

                # Calculate the new center coordinates
                new_center_i = n - 1
                new_center_j = m - 1

                # Calculate the starting indices in the new array
                start_i = new_center_i - i
                start_j = new_center_j - j

                # Place the original array in the new array
                new_array[start_i:start_i + n, start_j:start_j + m] = original_array

                return new_array
            center_mat = center_array(self.matrix, self.cur_y, self.cur_x)
            #print(center_mat.shape)
            return center_mat
    
    def _get_flat_vector(self):
        return self.flat_vector.reshape((self.num_flat_channels,))

    def validate_observation(self, observation):
        try:
            # Check action mask
            action_mask = observation['action_mask']
            if action_mask.shape != self.observation_space.spaces['action_mask'].shape:
                raise ValueError(f"Action mask shape mismatch: {action_mask.shape} vs {self.observation_space.spaces['action_mask'].shape}")
            if action_mask.dtype != bool:
                raise ValueError(f"Action mask dtype mismatch: {action_mask.dtype} vs bool")
        
            # Check observations
            obs_data = observation['matrix']
            expected_space = self.observation_space.spaces['matrix']
            if obs_data.shape != expected_space.shape:
                raise ValueError(f"shape mismatch: {obs_data.shape} vs {expected_space.shape}")
            if obs_data.dtype != expected_space.dtype:
                raise ValueError(f"dtype mismatch: {obs_data.dtype} vs {expected_space.dtype}")
            if np.any(obs_data < expected_space.low) or np.any(obs_data > expected_space.high):
                #raise ValueError(f"values out of bounds: expected between {expected_space.low} and {expected_space.high}")
                for i in range(obs_data.shape[0]):
                    for j in range(obs_data.shape[1]):
                        for k in range(obs_data.shape[2]):
                            if obs_data[i, j, k] < -1 or obs_data[i, j, k] > 1:
                                print("out of bounds", i, j, k, obs_data[i, j, k])
    
            # Check additional metrics
            for target in self.target_ranges:
                metric_data = observation[target.name]
                expected_space = self.observation_space.spaces[target.name]
                if metric_data.shape != expected_space.shape:
                    raise ValueError(f"{target} shape mismatch: {metric_data.shape} vs {expected_space.shape}")
                if metric_data.dtype != expected_space.dtype:
                    raise ValueError(f"{target} dtype mismatch: {metric_data.dtype} vs {expected_space.dtype}")
                if np.any(metric_data < expected_space.low) or np.any(metric_data > expected_space.high):
                    raise ValueError(f"{target} values out of bounds: expected between {expected_space.low} and {expected_space.high}")
        
            print("Observation is valid.")
        except ValueError as e:
            print(f"Validation error: {e}")
    
    def _get_obs(self):
        # bugfix: https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614/2
        obs = OrderedDict()

        kvs = [
            ("action_mask", self.valid_actions),
            ("matrix", self._get_matrix()),
            ("flat", self._get_flat_vector()),
            
            # metrics for tensorboard.
            *[(f'{target.name}', np.array([self.cur_values[target.name]], dtype=np.float32)) for target in self.target_ranges],
        ]
        kvs.sort()
        for k, v in kvs:
            obs[k] = v
        #obs.update({OrderedDict(kvs)})
        #self.validate_observation(obs)
        return obs

    def _get_info(self):
        return {}
     
    def get_cur_values(self):
        return {}
    
    def update_target_arrays(self):
        self.cur_values = self.get_cur_values()
        
        if self.weight_averages:
            assert False
            metric_stats = ray.get_actor("metrics")
            stats = ray.get(metric_stats.get.remote())
            if len(stats) == 0:
                normalized_metric_avgs = {target: 1/len(self.targets) for target in self.targets}
            else:
                metric_avgs = {target: stats[f'{target}_error_mean'] for target in self.targets}
                #metric_avgs['visual_novelty'] *= 2
                normalized_metric_avgs = {target: avg/sum(metric_avgs.values()) for target, avg in metric_avgs.items()}
                assert np.round(sum(normalized_metric_avgs.values()), 2) == 1
        else:
            normalized_metric_avgs = {target: 1/len(self.targets) for target in self.targets}
                
        for i, target in enumerate(self.target_ranges):
            if not (-1 <= self.cur_values[target.name] <= 10000):
                print(f"Error: {target.name} out of range: {self.cur_values[target.name]}")
            self.cur_values_mat[i] = self.cur_values[target.name]

        loss = 0
        data = []
        for i, target in enumerate(self.active_target_ranges):
            if target.min_val == target.max_val:
                if target.min_val == 0:
                    target_val = 1
                    assert 0 <= self.cur_values[target.name] <= 1, f"{target.name} out of range: {self.cur_values[target.name]}"
                    actual_val = 1 - self.cur_values[target.name]
                else:
                    target_val = 1 #MAX_VALUE
                    actual_val = self.cur_values[target.name] / target.max_val
            else:
                target_val = (self.targets[target.name] - target.min_val) / (target.max_val - target.min_val)
                actual_val = (self.cur_values[target.name] - target.min_val) / (target.max_val - target.min_val)
            actual_val = max(-MAX_VALUE, min(MAX_VALUE, actual_val))
            target_val = max(-MAX_VALUE, min(MAX_VALUE, target_val))
            
            error = abs(target_val - actual_val) * target.weight * (1 / len(self.active_target_ranges))
            loss += error 

        self.prev_loss = self.loss
        self.loss = loss
        #print(tabulate(data, headers=['#', 'property', 'target val', 'scaled', 'actual value', 'scaled', 'loss contribution']))
        
    def reset_ex(self):
        raise NotImplementedError
    
    def reset(self, *, seed=None, options=None, targets=None, target_weights=None):
        raise NotImplementedError
    
    def get_done_and_reward(self):
        raise NotImplementedError
    
    def apply_action(self, elmt, y, x):
        raise NotImplementedError
    
    def apply_action_with_number(self, action):
        raise NotImplementedError
    
    def step(self, action):
        assert self.valid_actions[action[0]], f"Invalid action sent to env: {action}"
    
        self.apply_action_with_number(action)
        self.cur_num_steps += 1
    
        self.update_target_arrays()
        self._fix_action_mask()
    
        terminated, truncated, reward = self.get_done_and_reward()
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
