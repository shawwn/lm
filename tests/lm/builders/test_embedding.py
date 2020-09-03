import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf
from absl.testing import parameterized
from tensorflow.python.framework import (  # pylint:disable=g-direct-tensorflow-import
    test_util,
)

import mock
from lm.builders import base, embeddings


class LayersTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        (4, True, "not_channels"), (8, False, "channels"),
    )
    def testDense(self, units, use_bias, new_dim_name):
        batch = 2
        channels = 3
        inputs = tf.random_normal([batch, channels])

        graph = mtf.Graph()
        mesh = mtf.Mesh(graph, "my_mesh")
        batch_dim = mtf.Dimension("batch", batch)
        channels_dim = mtf.Dimension("channels", channels)
        new_dim = mtf.Dimension(new_dim_name, units)

        mtf_inputs = mtf.import_tf_tensor(
            mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim])
        )
        mtf_outputs = mtf.layers.dense(
            mtf_inputs,
            new_dims=new_dim,
            reduced_dims=[channels_dim],
            activation=mtf.relu,
            use_bias=use_bias,
        )
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
            shape=[], layout={}, devices=[""]
        )
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

        expected_outputs = tf.keras.layers.Dense(
            units=units, activation=tf.nn.relu, use_bias=use_bias
        )(inputs)
        tf_group = lowering.copy_masters_to_slices()
        init = tf.global_variables_initializer()
        self.evaluate(init)
        self.evaluate(tf_group)
        actual, expected = self.evaluate([actual_outputs, expected_outputs])

        self.assertEqual(actual.shape, expected.shape)


class EmbeddingTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        (8,), (128,),
    )
    def testEmbedding(self, vocab_size):
        batch = 2
        seq = 3
        x = tf.random_normal([batch, seq])

        m = base.MeshGraph("test")
        m.batch_dim = 10

        self.assertEqual(m.batch_dim, mtf.Dimension(name="batch", size=10))
