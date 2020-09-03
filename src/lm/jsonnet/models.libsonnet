{
    GPT2(vocab_size=128) :: {
        kind: "neogpt.models.GPT2",
        n_ctx: 8,
        n_embd: 8,
        n_head: 8,
        n_vocab: vocab_size,
        n_layer: 2,
        scale_by_depth: false,
        scale_by_in: false,
        mesh_shape: "batch:1",
        layout: "batch:1",
        activation_function: "gelu",
        attention_types: [
            [["global"], self.n_layer],
        ],
        auto_layout: false,
        auto_layout_and_mesh_shape: false,
        stop_at_token: 2,
        remove_partial_sequences: true,
        scalenorm: true,
        no_weight_tie: false,
        regularization: {
            embed_dropout: 0.1,  // embedding dropout
            attn_dropout: 0.1,
            res_dropout:0.1,
        },
    }
}