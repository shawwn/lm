"this model configures the base mesh transformer for lm"
# from enum import Enum
# from typing import Callable, List, Optional

# import mesh_tensorflow as mtf
# from pydantic import BaseModel

# import lm


# class ModelType(Enum):
#     bitransformer = "bitransformer"
#     bi_student_teacher = "bi_student_teacher"
#     lm = "lm"
#     aligned = "aligned"


# class MeshTransformerModelConfig(BaseModel):
#     model_type: ModelType
#     vocabulary: tuple
#     mesh_shape: Callable
#     model_dir: str
#     autostack: bool
#     learning_rate_schedule: Optional[Callable]
#     keep_checkpoint_max: int
#     save_checkpoints_steps: int
#     optimizer: object
#     predict_fn: Optional[Callable]
#     variable_filter: str  # a variable will be trained only if matches this one
#     ensemble_inputs: int
#     use_tpu: str
#     tpu_job_name: str
#     iterations_per_loop: int
#     cluster: TPUClusterResolver
#     init_checkpoint: str
#     mesh_devices: List[str]  # for GPU


# @lm.register_model("lm.models.MeshTransformer")
# class MeshTransformerModel:
#     def __init__(self, config: MeshTransformerModelConfig):
#         self.config = config

#     def __call__(self):
#         in_vocab_size = inputs_vocabulary(vocabulary).vocab_size
#         out_vocab_size = targets_vocabulary(vocabulary).vocab_size
#         transformer_model = build_model(
#             model_type=sef.config.model_type,
#             input_vocab_size=in_vocab_size,
#             output_vocab_size=out_vocab,
#             layout_rules=self.config.layout_rules,
#             mesh_shape=self.config.mesh_shape,
#         )

#         model_fn = tpu_estimator_model_fn(
#             model_type=self.model_type,
#             transformer_model=self.transformer_model,
#             vocabulary=self.vocabulary,
#             model_dir=self.model_dir,
#             use_tpu=self.use_tpu,
#             mesh_shape=self.config.mesh_shape,
#             layout_rules=self.layout_rules,
#             batch_size=batch_size,
#             sequence_length=self.config.sequence_length,
#             autostack=self.config.autostack,
#             learning_rate_schedule=self.config.learning_rate_schedule,
#             keep_checkpoint_max=self.config.keep_checkpoint_max,
#             save_checkpoints_steps=self.config.save_checkpoints_steps,
#             optimizer=self.config.optimizer,
#             predict_fn=self.config.predict_fn,
#             variable_filter=self.config.variable_filter,
#             ensemble_inputs=ensemble_inputs,
#             init_checkpoint=init_checkpoint,
#             mesh_devices=mesh_devices,
#         )
