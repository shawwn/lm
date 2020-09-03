import abc
from typing import List

import mesh_tensorflow as mtf
from pydantic.dataclasses import dataclass

from . import base, nn


@dataclass
class InputBuilderConfig:
    vocab_size: int
    embedding_size: int
    sequence_length: int
    embedding_dropout: float


class InputBuilder:
    def __init__(self, mesh: base.MeshGraph, config: InputBuilderConfig):
        super().__init__(mesh)
        self.config = config

    def add_positional_embeddings(self):
        """
        returns [sequence_length, embedding_dim]
        """
        # vocab_size = self.config.vocab_size
        # sequence_length = self.config.sequence_length
        # embedding_dim = self.config.embedding_dim
        return self.get_variable(
            "pos_embeddings", [1, self.sequence_length, self.embedding_dim]
        )

    def add_tokens_embeddings(self, tokens):
        """
        tokens: [batch_size, seq_len, 1]
        """
        # vocab_size = self.config.vocab_size
        # embedding_dim = self.config.embedding_dim
        # embedding_table = self.add_var(
        #     "tok_embeddings", shape=(vocab_size, embedding_dim)
        # )
        # flat_input_ids = tf.reshape(tokens, [-1])
        # one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        # # [ vocab_size, embedding_dim ] * [vocab_size, embedding_dim]
        # output = tf.matmul(one_hot_input_ids, embedding_table)
        # output = tf.reshape(output, shape=(-1, sequence_length, embedding_dim))

        with tf.variable_scope("embeddings"):
            # Perform embedding lookup on the token ids.
            self.embedding_table = self.get_variable(
                "tokens_embeddings",
                [self.vocab_dim, self.model_dim],
                initializer=self.embedding_initializer,
            )
            self.tokens_embedding_output = mtf.gather(
                self.embedding_table, tokens, self.vocab_dim
            )
        return self.tokens_embedding_output

    def add_dropout(self, x):
        return super().add_dropout(x, dropout_prob=self.config.embedding_dropout)

    def __call__(self, tokens, batch_size, sequence_length):
        """
        tokens: [batch_size, sequence_length, 1]
        """

        self.batch_dim = self.add_dim("batch", batch_size)
        self.seq_dim = self.add_dim("seq", sequence_length)

        self.import_tf_tensor(tokens, [self.batch_dim, self.seq_dim])

        emb = self.add_tokens_embeddings(tokens)
        pos = self.add_positional_embeddings()
        x = emb + pos
        self.add_layer_norm_with_no_bias(x)
        return self.add_dropout(x)

    @property
    def vocab_dim(self):
        # pad vocab to a multiple of 128 so as to be splittable.
        n = self.config.vocab_size
        return mtf.Dimension("vocab", n + (-n % 128))

    @property
    def embedding_dim(self):
        return mtf.Dimension("embedding", self.config.embedding_size)

    @property
    def hidden_dim(self):
        "also known as the model dimension"
        return mtf.Dimension("hidden", self.config.d_model)
