from absl.testing import absltest

import lm.config
import lm.tasks.math
import lm.tf


class UnitTests(absltest.TestCase):
    def setUp(self):
        taskcfg = lm.config.load("etc/lm/tasks/addition.jsonnet")
        self.addition = lm.tasks.math.Addition(**taskcfg)
        return super().setUp()

    # def test_load_from_config(self):
    #     addition = self.addition
    #     self.assertTrue(addition.dataset)
    #     self.assertTrue(addition.dataset.vocab_size)
    #     self.assertTrue(addition.kind)
    #     self.assertTrue(addition.description)

    # def test_generate(self):
    #     infeed = self.addition.build_infeed()
    #     batch_size = 8
    #     for v in lm.tf.consume(infeed, params={'batch_size': batch_size}):
    #         self.assertEqual(v.shape, (batch_size, self.addition.context_length))
    #         break


if __name__ == "__main__":
    absltest.main()
