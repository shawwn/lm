from pydantic.datasclasses import dataclass


@dataclass
class ModelConfig:
    n_layer: int
    input_config: InputConfig
    transfomer_config: TransformerConfig


class ModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.add_transformer_block = TransformerBlockBuilder(
            self.config.transfomer_config
        )
        self.add_input = InputBuilder(config.input_config)

    def __call__(self, inputs, params):
        tokens = inputs["tokens"]
        x = self.add_input(tokens)
        for l in range(self.config.n_layer):
            x = self.add_transformer_block(f"layer_{l}", x)
        return x
