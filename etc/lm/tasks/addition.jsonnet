local infeeds = import "infeeds.libsonnet";

local AdditionProducer(seed=1337, ndigits=2) = {
    kind: "lm.datasets.AdditionProducer",
    seed: seed,
    ndigits: ndigits,
    context_length: ndigits * 3,
    vocab_size: 10,
};

local Source() = {
    kind: "generator",
};

{
    kind: "lm.tasks.Addition",
    description: "sequence to sequence",
    dataset: AdditionProducer(),
    infeed: infeeds.TFRecordDatasetReader(source={})
}