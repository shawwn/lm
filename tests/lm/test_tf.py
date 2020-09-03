import numpy as np
import tensorflow as tf
from absl.testing import absltest

import lm.examples
import lm.tf


class TestTF(absltest.TestCase):
    def test_as_dataset(self):
        def infeed(params):
            def simplegen():
                for i in range(batch_size):
                    yield lm.examples.Seq2SeqSimpleExample(
                        np.ones(8, dtype=np.int64) * i, np.zeros(8, dtype=np.int64)
                    ).serialize()

            ds = lm.tf.from_generator(lambda: simplegen)
            ds = ds.batch(params["batch_size"])
            return ds

        batch_size = 8
        for ex in lm.tf.consume(infeed, params=dict(batch_size=batch_size)):
            self.assertEqual(ex.shape, (batch_size,))
            break


if __name__ == "__main__":
    absltest.main()
