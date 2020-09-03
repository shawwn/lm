local infeeds = import "infeeds.libsonnet";
local datasets = import "datasets.libsonnet";

local SumOneDataset(max_sequence_length=8, vocab_size=10) = {
    kind: "lm.datasets.SumOne",
    max_sequence_length: max_sequence_length,
    vocab_size: vocab_size,
    seed: 1337
};

{
    kind: "lm.tasks.SumOne",
    description: "sample task to learn to sum one to each input token",
    dataset: SumOneDataset(),
    infeed: infeeds.TFRecordDatasetReader(source=self.dataset),
}