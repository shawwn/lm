"Collection of Attention Layers"
import mesh_tensorflow as mtf
from pydantic.dataclasses import dataclass


@dataclass
class MultiHeadAttentionConfig:
    src_seq_len: int
    dst_seq_len: int
    n_head: int
    m_dim: int
    q_dim: int
    k_dim: int
    v_dim: int
    o_dim: int
    dtype: str


class MultiHeadAttentionBuilder:
    def __init__(self, config: MultiHeadAttentionConfig):
        self.config = config

    def add_random_uniform(self):
        return mtf.random_uniform_initializer(shape=shape)

    def add_var(self, name, *shape):
        return mtf.get_variable(name, shape=shape, initializer=create_initializer(),)

    def add_triangular_mask(self):
        """
        language model next token prediction mask
        returns: [batch, heads, dst_seq, src_seq]
        """
        nd = self.config.dst_seq_len
        ns = self.config.src_seq_len
        # add one dimension
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        mask = i >= (j - ns + nd)

        mask = tf.reshape(mask, (1, 1, nd, ns))

        return tf.cast(mask, self.config.dtype)

    def __call__(self, x):
        """
        x: [batch_dim, sequence, embedding_dim]
        """
        d = self.config.src_seq_len
        n_head = self.config.n_head
        m_dim = self.config.m_dim
        q_dim = self.config.q_dim
        k_dim = self.config.k_dim
        v_dim = self.config.v_dim
        o_dim = self.config.o_dim

        self.mask = self.add_triangular_mask()  # [batch, heads, dest, src]
        self.M = self.add_var("M", 1, m_dim, d)
        self.Q = self.add_var("Q", n_head, d, q_dim)
        self.K = self.add_var("K", n_head, d, k_dim)
        self.V = self.add_var("V", n_head, d, v_dim)
        self.O = self.add_var("O", n_head, d, o_dim)

        return batch_query_multi_head_attention(
            x, self.M, self.mask, self.Q, self.K, self.V, self.O
        )
