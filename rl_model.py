
import numpy as np
import tensorflow as tf
from gymnasium.spaces import Box, Dict
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from gen_envs.base import MAX_VALUE
from complex_net import ComplexInputNetworkMultipleShapes

# https://github.com/PathmindAI/nativerl/blob/f43e54486992e366c1b40c95d9da308ec9df7713/nativerl/python/pathmind_training/models.py
class TargetGridActionMaskModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, size, env, *args, **kwargs):
        # filter sizes of 1x1,2x2,4x4 are bad
        # ref: https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15
        model_config["use_lstm"] = False
        model_config["vf_share_layers"] = True
        model_config["conv_filters"] = {}
        model_config["conv_filters"][size] = [
            [32, [3, 3], 2],
            [32, [3, 3], 2],
            [32, [3, 3], 2],
            [32, [3, 3], 2],
        ]
        assert size in model_config["conv_filters"]
        model_config["post_fcnet_hiddens"] = [64, 64, 64] # ***
        model_config["num_rides"] = len(env.ride_types)
        model_config["ranking_loss_weight"] = 2 #0.5
        
        self.size = size
        super().__init__(obs_space, action_space, num_outputs, model_config, name, *args, **kwargs)

        print("Channels:", env.num_grid_channels, env.num_flat_channels) # 57 176 
        self.action_embed_model = ComplexInputNetworkMultipleShapes(
            Dict({
                'obs': Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(*size, env.num_grid_channels), dtype=np.float32),
                'flat': Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(env.num_flat_channels,), dtype=np.float32),
            }),
            action_space, model_config, name + "_action_embedding"
        )
    
    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        obs = input_dict["obs"]["matrix"]
        flat = input_dict["obs"]["flat"]
        #print("Shapes:", obs.shape, flat.shape)
        logits_list, _ = self.action_embed_model({"obs": {"matrix": obs, "flat": flat}})
        
        masked_rides = 'ride_action_mask_0' in input_dict['obs']
        masked_logits_list = []
        for i, logits in enumerate(logits_list):
            # Apply masking to each set of logits
            
            if i == 0:
                inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
                masked_logits = logits + inf_mask
            elif masked_rides and (i > 0 and i < len(logits_list) - 2):
                ride_action_mask = input_dict['obs'][f'ride_action_mask_{i-1}']
                inf_mask = tf.maximum(tf.math.log(ride_action_mask), tf.float32.min)
                #print(logits.shape, inf_mask.shape)
                masked_logits = logits + inf_mask
            else:
                masked_logits = logits
            masked_logits_list.append(masked_logits)

        # Combine masked logits back into a list or tensor (depending on your framework)
        tensors = tf.concat(masked_logits_list, axis=-1)
        return tensors, state
 
    def value_function(self):
        return self.action_embed_model.value_function()
