"stateless layer functions with minimal configuration"
import mesh_tensorflow as mtf


# from mesh_tensorflow bert
def layer_norm(
    x,
    dim: mtf.Dimension,
    epsilon: float = 1e-6,
    subtract_mean=True,
    use_scale=True,
    use_bias=True,
    name=None,
):
    """Layer normalization over dimension dim.

    Args:
        x: a mtf.Tensor whose shape contains dim.
        dim: a mtf.Dimension
        epsilon: a floating point number
        subtract_mean: a boolean
        use_scale: a boolean
        use_bias: a boolean
        name: a string used for tf.variable_scope.

    Returns:
        a mtf.Tensor with same shape as x.
    """
    with tf.variable_scope(name, default_name="layer_norm"):
        if subtract_mean:
            x -= mtf.reduce_mean(x, reduced_dim=dim)
        variance = mtf.reduce_mean(mtf.square(x), reduced_dim=dim)
        x *= mtf.rsqrt(variance + epsilon)
        if use_scale:
            x *= mtf.get_variable(
                x.mesh,
                "scale",
                mtf.Shape([dim]),
                initializer=tf.ones_initializer(),
                activation_dtype=x.dtype,
            )
        if use_bias:
            x += mtf.get_variable(
                x.mesh,
                "bias",
                mtf.Shape([dim]),
                initializer=tf.zeros_initializer(),
                activation_dtype=x.dtype,
            )
        return x
