local infeeds = import "infeeds.libsonnet";

local SumOneProducer(seed=1337, ndigits=2) = {
    kind: "lm.datasets.SumOneGen",
    seed: seed,
    ndigits: ndigits,
    context_length: ndigits * 3,
    vocab_size: 10,
};

{
    kind: "lm.tasks.SumOne",
    description: "sample task to learn to sum one to each input token",
    dataset: SumOneProducer(seed=1337, ndigits=2),
    infeed: infeeds.ExampleGenerator(producer=self.dataset)
}