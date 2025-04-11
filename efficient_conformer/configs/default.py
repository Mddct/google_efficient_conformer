import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.output_size = 256
    config.attention_heads = 4
    config.linear_units = 2048
    config.dropout_rate = 0.1
    config.positional_dropout_rate = 0.1
    config.attention_dropout_rate = 0.0
    config.normalize_before = True
    config.query_bias = True
    config.key_bias = True
    config.value_bias = True
    config.activation_type = "relu"
    config.gradient_checkpointing = False
    config.use_sdpa = False
    config.layer_norm_type = "rms_norm"
    config.norm_eps = 1e-5
    config.n_kv_head = None
    config.head_dim = None
    config.selfattention_layer_type = "selfattn"
    config.mlp_type = "position_wise_feed_forward"  # ['position_wise_feed_forward', 'moe', 'gated']
    config.mlp_bias = True
    config.n_expert = 8
    config.n_expert_activated = 2
    config.right_context = 2
    config.left_context = 15

    # total blocks: first_n_layers + num_blocks
    config.first_n_layers = 3
    config.num_blocks = 6
    config.causal = True
    config.cnn_module_kernel = 15
    config.use_cnn_module = True
    config.final_norm = False
    config.cnn_module_norm = "batch_norm"
    config.conv_bias = True
    config.conv_norm_eps = 1e-6
    config.conv_inner_factor = 2
    config.macaron_style = True

    return config
