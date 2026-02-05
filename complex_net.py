import tree  # pip install dm_tree
import pickle
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.tf_utils import one_hot

from visionnet2d import VisionNetwork2D

tf1, tf, tfv = try_import_tf()

class ComplexInputNetworkMultipleShapes(TFModelV2):
    """TFModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and vaulue heads.
    """

    def __init__(self, obs_space, action_space, model_config, name):
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )
        
        num_outputs = action_space['position_and_type'].nvec

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )
        super().__init__(
            self.original_space, action_space, num_outputs, model_config, name
        )

        self.flattened_input_space = flatten_space(self.original_space)

        # Build the CNN(s) given obs_space's image components.
        self.cnns = {}
        self.one_hot = {}
        self.flatten_dims = {}
        self.flatten = {}
        concat_size = 0
        
        cnn_num = 0
        for i, component in enumerate(self.flattened_input_space):
            # Image space.
            if len(component.shape) == 3:
                config = {
                    "conv_filters": model_config["conv_filters"][component.shape[:-1]]
                    if "conv_filters" in model_config
                    else get_filter_config(component.shape),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                    "vf_share_layers": True,
                }
                self.cnns[i] = VisionNetwork2D( #
                    component,
                    action_space=action_space,
                    num_outputs=None,
                    model_config=config,
                    name="cnn_{}".format(i),
                )
                concat_size += self.cnns[i].num_outputs
                cnn_num += 1
            elif len(component.shape) == 4:
                config = {
                    "conv_filters": model_config["conv_filters"][component.shape[:-1]]
                    if "conv_filters" in model_config
                    else get_filter_config(component.shape),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                    "vf_share_layers": True,
                }
                self.cnns[i] = VisionNetwork3D( #
                    component,
                    action_space=action_space,
                    num_outputs=None,
                    model_config=config,
                    name="cnn_{}".format(i),
                )
                concat_size += self.cnns[i].num_outputs
                cnn_num += 1
            # Discrete|MultiDiscrete inputs -> One-hot encode.
            elif isinstance(component, (Discrete, MultiDiscrete)):
                if isinstance(component, Discrete):
                    size = component.n
                else:
                    size = np.sum(component.nvec)
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.one_hot[i] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="tf",
                    name="one_hot_{}".format(i),
                )
                concat_size += self.one_hot[i].num_outputs
            # Everything else (1D Box).
            else:
                size = int(np.product(component.shape))
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.flatten[i] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="tf",
                    name="flatten_{}".format(i),
                )
                self.flatten_dims[i] = size
                concat_size += self.flatten[i].num_outputs

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation", "relu"),
        }
        print("Post FC stack config:", post_fc_stack_config)
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"), float("inf"), shape=(int(concat_size),), dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="tf",
            name="post_fc_stack",
        )

        # Actions and value heads.
        self.logits_and_value_model = None
        self._value_out = None
        self._ride_ranking = None
        if num_outputs is not None:
            # Action-distribution head.
            concat_layer = tf.keras.layers.Input((self.post_fc_stack.num_outputs,))

            logits_layers = []
            for i, n in enumerate(action_space['position_and_type'].nvec):
                logits_layer = tf.keras.layers.Dense(
                    n,
                    activation=None,
                    kernel_initializer=normc_initializer(0.01),
                    name=f"logits_{i}",
                )(concat_layer)
                logits_layers.append(logits_layer)
            
            logits_layer_price = tf.keras.layers.Dense(
                2,
                activation=None,
                kernel_initializer=normc_initializer(0.01),
                name=f"logits_price",
            )(concat_layer)
            logits_layers.append(logits_layer_price)

            logits_layer_queues = tf.keras.layers.Dense(
                5,
                activation=None,
                kernel_initializer=normc_initializer(0.01),
                name="logits_queue_line_lengths",
            )(concat_layer)
            logits_layers.append(logits_layer_queues)
            
            # Create the value branch model.
            value_layer = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=normc_initializer(0.01),
                name="value_out",
            )(concat_layer)
            
            self.logits_and_value_model = tf.keras.models.Model(
                concat_layer, [logits_layers, value_layer] #, ride_ranking_layer]
            )
            print("Logits and value model:")
            self.logits_and_value_model.summary()

            print("CNNs:", self.cnns)
            print("Flatten:", self.flatten)
            print("One-hot:", self.one_hot)
            print("Post FC stack:", self.post_fc_stack)
        else:
            self.num_outputs = self.post_fc_stack.num_outputs

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="tf"
            )
        # Push images through our CNNs.
        outs = []
        for i, component in enumerate(tree.flatten(orig_obs)):
            if i in self.cnns:
                cnn_out, _ = self.cnns[i](SampleBatch({SampleBatch.OBS: component}))
                outs.append(cnn_out)
            elif i in self.one_hot:
                if "int" in component.dtype.name:
                    one_hot_in = {
                        SampleBatch.OBS: one_hot(
                            component, self.flattened_input_space[i]
                        )
                    }
                else:
                    one_hot_in = {SampleBatch.OBS: component}
                one_hot_out, _ = self.one_hot[i](SampleBatch(one_hot_in))
                outs.append(one_hot_out)
            else:
                nn_out, _ = self.flatten[i](
                    SampleBatch(
                        {
                            SampleBatch.OBS: tf.cast(
                                tf.reshape(component, [-1, self.flatten_dims[i]]),
                                tf.float32,
                            )
                        }
                    )
                )
                outs.append(nn_out)
        # Concat all outputs and the non-image inputs.
        out = tf.concat(outs, axis=-1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(SampleBatch({SampleBatch.OBS: out}))

        # No logits/value branches.
        if not self.logits_and_value_model:
            return out, []

        # Logits- and value branches.
        logits, values = self.logits_and_value_model(out)
        self._value_out = tf.reshape(values, [-1])
        #self._ride_ranking = ride_ranking
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out

    def pairwise_ranking_loss(self, scores, targets, margin=1.0):
        """
        Computes a pairwise hinge ranking loss.
        
        Args:
            scores: Tensor of shape [B, num_rides] with model's ride preference scores.
            targets: Tensor of shape [B, num_rides] with ground-truth ranking values.
                     For each ride pair (i, j), if targets[:, i] > targets[:, j],
                     then we want scores[:, i] to exceed scores[:, j] by at least `margin`.
            margin: The required margin between scores.
        
        Returns:
            A scalar tensor representing the average ranking loss.
        """
        B = tf.shape(scores)[0]
        if len(targets.shape) == 1:
            targets = tf.expand_dims(targets, 0)
            targets = tf.tile(targets, [B, 1])

        #targets = np.tile(targets, (B, 1))
        N = self.num_rides
        loss = 0.0
        # Loop over all ride pairs. (Vectorizing this may be more efficient in practice.)
        for i in range(N):
            for j in range(N):
                # Consider only pairs where the ground-truth ranking for ride i is higher than ride j.
                mask = tf.cast(tf.greater(targets[:, i], targets[:, j]), tf.float32)
                diff = scores[:, i] - scores[:, j]
                loss += tf.reduce_mean(mask * tf.nn.relu(margin - diff))
        return loss
