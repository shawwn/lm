"Block utils library"
import mesh_tensorflow as mtf


@dataclass
class TransformerConfig:
    dst_seq_len: int
    multi_head_attention_config: MultiHeadAttentionConfig


class TransformerBlockBuilder:
    def __init__(self, config: TransformerConfig):
        self._config = config
        self.multi_head_attention = MultiHeadAttentionBuilder(
            config.multi_head_attention_config
        )

    def add_layer_norm(self, inputs):
        return tf.contrib.layers.layer_norm(
            inputs,
            center=True,
            scale=True,
            activation_fn=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            begin_norm_axis=1,
            begin_params_axis=-1,
            scope=None,
        )

    def add_feed_forward(self, inputs):
        return tf.layers.dense(
            inputs,
            units=self._config.dst_seq_len,
            activation=tf.nn.relu,
            use_bias=True,
        )

    def __call__(self, name, x):
        with tf.variable_scope(name):
            x = x + self.add_layer_norm(self.multi_head_attention(x))
            x = x + self.add_layer_norm(self.add_feed_forward(x))
            return
