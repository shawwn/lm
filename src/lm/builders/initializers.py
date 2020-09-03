import mesh_tensorflow as mtf


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


class Initializer:
    def __call__(self):
        pass

    @property
    def dense_initializer(self):
        if self.config.initializer_range:
            return tf.truncated_normal_initializer(stddev=self.config.initializer_range)
        else:
            return mtf.layers.VarianceScalingInitializer(scale=0.4)

    @property
    def embedding_initializer(self):
        initializer = self.dense_initializer
        if isinstance(initializer, mtf.layers.DenseInitializer):
            # embedding matrix is also used as classifier weight matrix.
            # scale it appropriately.
            return initializer(reduced_dims=[self.model_dim], new_dims=[self.vocab_dim])
        else:
            return initializer
