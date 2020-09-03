local optimizers = import 'optimizers.libsonnet';
local models = import 'models.libsonnet';

local lr() = {
   lr: 0.0001,
   lr_decay: "cosine",
   warmup_steps: 0,
};

local Schedule() = {
   steps: self.steps_per_iteration * 10,                // total number of steps to run
   steps_per_iteration: 1000,  // how many steps to loop on-device
   steps_per_checkpoint: self.steps_per_iteration, // save a checkpoint after this num of steps
};

local Trainer() = {
   device: {},
   task: {},
   infeed: {},
   model: models.GPT2(),
   model_path: "/tmp/checkpoints/",
   schedule: Schedule(),
   gradients: {
      optimizer: optimizers.Adam(),
      learning_rate: lr(),
      weight_decay: 0.1,
      gradient_clipping: 0.5,
   },
};

Trainer() // main configuration