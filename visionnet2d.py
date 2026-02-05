import gymnasium as gym
from typing import Dict, List, Optional, Sequence

from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType

tf1, tf, tfv = try_import_tf()


# ============================================================================
# CBAM: Convolutional Block Attention Module
# Paper: https://arxiv.org/abs/1807.06521
# ============================================================================

def channel_attention(input_feature, ratio=8, name_prefix=""):
    """Modulo di Attenzione di Canale.
    
    Impara quali canali (feature) sono più importanti pesandoli dinamicamente.
    """
    channel = input_feature.shape[-1]
    
    # Shared MLP per avg e max pool
    shared_dense1 = tf.keras.layers.Dense(
        max(channel // ratio, 1),
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=True,
        name=f"{name_prefix}channel_dense1"
    )
    shared_dense2 = tf.keras.layers.Dense(
        channel,
        kernel_initializer='he_normal',
        use_bias=True,
        name=f"{name_prefix}channel_dense2"
    )
    
    # Average pooling branch
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense1(avg_pool)
    avg_pool = shared_dense2(avg_pool)
    
    # Max pooling branch
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense1(max_pool)
    max_pool = shared_dense2(max_pool)
    
    # Combine e sigmoid
    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
    
    return tf.keras.layers.Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature, kernel_size=7, name_prefix=""):
    """Modulo di Attenzione Spaziale.
    
    Impara DOVE guardare nella mappa, creando una maschera di attenzione spaziale.
    """
    # Average e Max lungo l'asse dei canali
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    
    # Conv per generare la maschera spaziale
    cbam_feature = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False,
        name=f"{name_prefix}spatial_conv"
    )(concat)
    
    return tf.keras.layers.Multiply()([input_feature, cbam_feature])


def cbam_block(input_feature, ratio=8, name_prefix="cbam_"):
    """Blocco CBAM completo: Channel Attention → Spatial Attention."""
    x = channel_attention(input_feature, ratio, name_prefix)
    x = spatial_attention(x, kernel_size=7, name_prefix=name_prefix)
    return x


# ============================================================================
# VisionNetwork2D con CBAM
# ============================================================================

@DeveloperAPI
class VisionNetwork2D(TFModelV2):
    """Vision network con modulo opzionale di Attenzione (CBAM).
    
    Configurazione via custom_model_config:
        use_attention: bool (default True) - Attiva/disattiva CBAM
        attention_after_layer: int (default 2) - Dopo quale conv layer applicare CBAM
        attention_ratio: int (default 8) - Ratio di compressione nel channel attention
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        super(VisionNetwork2D, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        
        # --- Configurazione CBAM ---
        custom_config = model_config.get("custom_model_config", {})
        self.use_attention = custom_config.get("use_attention", True)
        self.attention_after_layer = custom_config.get("attention_after_layer", 2)
        self.attention_ratio = custom_config.get("attention_ratio", 8)
        # ---------------------------

        activation = get_activation_fn(
            self.model_config.get("conv_activation"), framework="tf"
        )
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="tf"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        input_shape = obs_space.shape
        self.data_format = "channels_last"

        inputs = tf.keras.layers.Input(shape=input_shape, name="matrix")
        last_layer = inputs
        self.last_layer_is_flattened = False

        # --- Stack Convoluzionale con CBAM ---
        attention_applied = False
        
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="same",
                data_format="channels_last",
                name="conv{}".format(i),
            )(last_layer)
            
            # Applicazione CBAM dopo il layer specificato
            if self.use_attention and not attention_applied and i >= self.attention_after_layer:
                print(f"[VisionNetwork2D] Adding CBAM Attention Block after conv{i}")
                last_layer = cbam_block(last_layer, ratio=self.attention_ratio, name_prefix=f"cbam_conv{i}_")
                attention_applied = True

        out_size, kernel, stride = filters[-1]

        # Ultimo layer conv
        if no_final_linear and num_outputs:
            last_layer = tf.keras.layers.Conv2D(
                out_size if post_fcnet_hiddens else num_outputs,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv_out",
            )(last_layer)
            
            layer_sizes = post_fcnet_hiddens[:-1] + ([num_outputs] if post_fcnet_hiddens else [])
            feature_out = last_layer

            for i, out_size in enumerate(layer_sizes):
                feature_out = last_layer
                last_layer = tf.keras.layers.Dense(
                    out_size,
                    name="post_fcnet_{}".format(i),
                    activation=post_fcnet_activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
        else:            
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv{}".format(len(filters)),
            )(last_layer)
            
            # Deconvoluzioni per espandere il campo recettivo (come nel codice originale)
            last_layer = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='valid', activation='relu', output_padding=0)(last_layer)
            last_layer = tf.keras.layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='valid', activation='relu', output_padding=0)(last_layer)
            last_layer = tf.keras.layers.Conv2DTranspose(32, kernel_size=1, strides=2, padding='valid', activation='relu', output_padding=0)(last_layer)
            last_layer = tf.keras.layers.Conv2DTranspose(1, kernel_size=2, strides=2, padding='valid', activation=None, output_padding=1)(last_layer)

            if num_outputs:
                if post_fcnet_hiddens:
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        post_fcnet_hiddens[0],
                        [1, 1],
                        activation=post_fcnet_activation,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out",
                    )(last_layer)
                    for i, out_size in enumerate(post_fcnet_hiddens[1:] + [num_outputs]):
                        feature_out = last_layer
                        last_layer = tf.keras.layers.Dense(
                            out_size,
                            name="post_fcnet_{}".format(i + 1),
                            activation=post_fcnet_activation if i < len(post_fcnet_hiddens) - 1 else None,
                            kernel_initializer=normc_initializer(1.0),
                        )(last_layer)
                else:
                    feature_out = last_layer
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        num_outputs,
                        [1, 1],
                        activation=None,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out",
                    )(last_layer)

                if last_cnn.shape[1] != 1 or last_cnn.shape[2] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, 1, "
                        "1, {} (`num_outputs`)] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the dims 1 and 2 "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(last_cnn.shape),
                        )
                    )
            else:                
                self.last_layer_is_flattened = True
                last_layer = tf.keras.layers.Flatten(data_format="channels_last")(last_layer)

                for i, out_size in enumerate(post_fcnet_hiddens):
                    last_layer = tf.keras.layers.Dense(
                        out_size,
                        name="post_fcnet_{}".format(i),
                        activation=post_fcnet_activation,
                        kernel_initializer=normc_initializer(1.0),
                    )(last_layer)
                feature_out = last_layer
                self.num_outputs = last_layer.shape[1]
                
        logits_out = last_layer

        # --- Value Branch (senza CBAM per velocità) ---
        if vf_share_layers:            
            if not self.last_layer_is_flattened:
                feature_out = tf.keras.layers.Lambda(
                    lambda x: tf.squeeze(x, axis=[1, 2])
                )(feature_out)
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01),
            )(feature_out)
        else:
            # Branch parallelo separato per il value
            vl_layer = inputs
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                vl_layer = tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                    activation=activation,
                    padding="same",
                    data_format="channels_last",
                    name="conv_value_{}".format(i),
                )(vl_layer)
            out_size, kernel, stride = filters[-1]
            vl_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv_value_{}".format(len(filters)),
            )(vl_layer)
            vl_layer = tf.keras.layers.Conv2D(
                1,
                [1, 1],
                activation=None,
                padding="same",
                data_format="channels_last",
                name="conv_value_out",
            )(vl_layer)
            value_out = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(vl_layer)

        self.base_model = tf.keras.Model(inputs, [logits_out, value_out])
        print("[VisionNetwork2D] Model with CBAM Attention built successfully!")
        print(f"[VisionNetwork2D] use_attention={self.use_attention}, after_layer={self.attention_after_layer}")
        # self.base_model.summary()  # Decommentare per debug

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"]
        if self.data_format == "channels_first":
            obs = tf.transpose(obs, [0, 2, 3, 1])
        model_out, self._value_out = self.base_model(tf.cast(obs, tf.float32))
        if self.last_layer_is_flattened:
            return model_out, state
        else:
            return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])


# ============================================================================
# Keras_VisionNetwork (Legacy, senza CBAM per compatibilità)
# ============================================================================

@DeveloperAPI
class Keras_VisionNetwork(tf.keras.Model if tf else object):
    """Generic vision network implemented in tf keras.

    An additional post-conv fully connected stack can be added and configured
    via the config keys:
    `post_fcnet_hiddens`: Dense layer sizes after the Conv2D stack.
    `post_fcnet_activation`: Activation function to use for this FC stack.
    """

    def __init__(
        self,
        input_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int] = None,
        *,
        name: str = "",
        conv_filters: Optional[Sequence[Sequence[int]]] = None,
        conv_activation: Optional[str] = None,
        post_fcnet_hiddens: Optional[Sequence[int]] = (),
        post_fcnet_activation: Optional[str] = None,
        no_final_linear: bool = False,
        vf_share_layers: bool = False,
        free_log_std: bool = False,
        **kwargs,
    ):

        super().__init__(name=name)

        if not conv_filters:
            conv_filters = get_filter_config(input_space.shape)
        assert len(conv_filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        conv_activation = get_activation_fn(conv_activation, framework="tf")
        post_fcnet_activation = get_activation_fn(post_fcnet_activation, framework="tf")

        input_shape = input_space.shape
        self.data_format = "channels_last"

        inputs = tf.keras.layers.Input(shape=input_shape, name="matrix")
        last_layer = inputs
        self.last_layer_is_flattened = False

        for i, (out_size, kernel, stride) in enumerate(conv_filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=conv_activation,
                padding="same",
                data_format="channels_last",
                name="conv{}".format(i),
            )(last_layer)

        out_size, kernel, stride = conv_filters[-1]

        if no_final_linear and num_outputs:
            last_layer = tf.keras.layers.Conv2D(
                out_size if post_fcnet_hiddens else num_outputs,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=conv_activation,
                padding="valid",
                data_format="channels_last",
                name="conv_out",
            )(last_layer)
            layer_sizes = post_fcnet_hiddens[:-1] + ([num_outputs] if post_fcnet_hiddens else [])
            for i, out_size in enumerate(layer_sizes):
                last_layer = tf.keras.layers.Dense(
                    out_size,
                    name="post_fcnet_{}".format(i),
                    activation=post_fcnet_activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
        else:
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=conv_activation,
                padding="valid",
                data_format="channels_last",
                name="conv{}".format(len(conv_filters)),
            )(last_layer)

            if num_outputs:
                if post_fcnet_hiddens:
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        post_fcnet_hiddens[0],
                        [1, 1],
                        activation=post_fcnet_activation,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out",
                    )(last_layer)
                    for i, out_size in enumerate(post_fcnet_hiddens[1:] + [num_outputs]):
                        last_layer = tf.keras.layers.Dense(
                            out_size,
                            name="post_fcnet_{}".format(i + 1),
                            activation=post_fcnet_activation if i < len(post_fcnet_hiddens) - 1 else None,
                            kernel_initializer=normc_initializer(1.0),
                        )(last_layer)
                else:
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        num_outputs,
                        [1, 1],
                        activation=None,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out",
                    )(last_layer)

                if last_cnn.shape[1] != 1 or last_cnn.shape[2] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, 1, "
                        "1, {} (`num_outputs`)] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the dims 1 and 2 "
                        "are both 1.".format(
                            conv_filters,
                            num_outputs,
                            list(last_cnn.shape),
                        )
                    )
            else:
                self.last_layer_is_flattened = True
                last_layer = tf.keras.layers.Flatten(data_format="channels_last")(last_layer)

                for i, out_size in enumerate(post_fcnet_hiddens):
                    last_layer = tf.keras.layers.Dense(
                        out_size,
                        name="post_fcnet_{}".format(i),
                        activation=post_fcnet_activation,
                        kernel_initializer=normc_initializer(1.0),
                    )(last_layer)
        logits_out = last_layer

        if vf_share_layers:
            if not self.last_layer_is_flattened:
                last_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01),
            )(last_layer)
        else:
            last_layer = inputs
            for i, (out_size, kernel, stride) in enumerate(conv_filters[:-1], 1):
                last_layer = tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                    activation=conv_activation,
                    padding="same",
                    data_format="channels_last",
                    name="conv_value_{}".format(i),
                )(last_layer)
            out_size, kernel, stride = conv_filters[-1]
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=conv_activation,
                padding="valid",
                data_format="channels_last",
                name="conv_value_{}".format(len(conv_filters)),
            )(last_layer)
            last_layer = tf.keras.layers.Conv2D(
                1,
                [1, 1],
                activation=None,
                padding="same",
                data_format="channels_last",
                name="conv_value_out",
            )(last_layer)
            value_out = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)

        self.base_model = tf.keras.Model(inputs, [logits_out, value_out])

    def call(
        self, input_dict: SampleBatch
    ) -> (TensorType, List[TensorType], Dict[str, TensorType]):
        obs = input_dict["obs"]
        if self.data_format == "channels_first":
            obs = tf.transpose(obs, [0, 2, 3, 1])
        model_out, self._value_out = self.base_model(tf.cast(obs, tf.float32))
        state = [v for k, v in input_dict.items() if k.startswith("state_in_")]
        extra_outs = {SampleBatch.VF_PREDS: tf.reshape(self._value_out, [-1])}
        if self.last_layer_is_flattened:
            return model_out, state, extra_outs
        else:
            return tf.squeeze(model_out, axis=[1, 2]), state, extra_outs
