
local AdditionGenerator = function(
                                ndigits=3, 
                                seed=1337) {
        kind: 'lm.infeeds.Generator',
        seed: seed,
        ndigits: ndigits,
        context_length: ndigits * 3,
        vocab_size: 10,
    };

{
    kind: "lm.tasks.Addition",
    description: "sequence to sequence",
    dataset: AdditionGenerator(ndigits=3)
}