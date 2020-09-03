import abc
from typing import List

import mesh_tensorflow as mtf
from pydantic.dataclasses import dataclass

from . import nn


class MeshGraph:
    def __init__(self, name="my_mesh"):
        self.graph = mtf.Graph()
        self._mesh = mtf.Mesh(self.graph, name)
        self.__dimensions = {}

    def get_variable(self, name: str, dims: List[mtf.Dimension], initializer=None):
        initializer = (
            initializer if initializer else self._default_initializer_for_variable(name)
        )
        return mtf.get_variable(
            self._mesh, name, mtf.Shape(dims), initializer=initializer,
        )

    def add_dropout(self, x, dropout_prob=0.0):
        return mtf.dropout(x, keep_prob=1.0 - dropout_prob)

    def add_layer_norm_with_bias(self, x, dim: mtf.Dimension):
        return nn.layer_norm(x, dim, subtract_mean=True, use_bias=True,)

    def add_layer_norm_no_bias(self, x, dim: mtf.Dimension):
        return nn.layer_norm(x, dim, subtract_mean=False, use_bias=False,)

    def _default_initializer_for_variable(self, name):
        return mtf.layers.VarianceScalingInitializer(scale=0.4)

    def __getattr__(self, name):
        if name.endswith("_dim"):
            v = self.__dimensions.get(name, None)
            if v is not None:
                return v

    def __setattr__(self, name: str, value):
        if name.endswith("_dim"):
            self.__dimensions[name] = mtf.Dimension(name[: -len("_dim")], value)
            return
        super().__setattr__(name, value)

    def import_tf_tensor(self, x, dims):
        return mtf.import_tf_tensor(self.mesh, x, shape=mtf.Shape(dims))
