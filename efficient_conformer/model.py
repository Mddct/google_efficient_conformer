from typing import Optional, Sequence, Tuple

import torch
import torch.utils.checkpoint as ckpt
from torch.nn import init
from wenet.transformer.attention import T_CACHE
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_MLP_CLASSES, WENET_NORM_CLASSES)
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import causal_or_lookahead_mask, make_pad_mask


# https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/modular_gemma3n.py
class Gemma3nAudioCumulativeGroupNorm(torch.nn.Module):
    """Applies Group Normalization cumulatively over the time dimension.

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    """

    def __init__(
        self,
        num_channels: int,  # Number of channels (size of the last dimension)
        feature_dims: Sequence[
            int],  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps

        # Scale parameter depends only on the channel dimension
        self.weight = torch.nn.Parameter(torch.ones(num_channels))

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies cumulative group norm, optionally using a mask.

        Args:
          hidden_states: Input tensor, shape [B, T, *feature_dims, C].

        Returns:
          Normalized tensor with the same shape as x.
        """
        expected_input_suffix = self.feature_dims + (self.num_channels, )
        if hidden_states.shape[2:] != expected_input_suffix:
            raise ValueError(
                f"Input tensor shape suffix {hidden_states.shape[2:]} does not match expected"
                f" suffix (feature_dims + num_channels) {expected_input_suffix}"
            )

        input_dtype = hidden_states.dtype
        # Calculations are performed in float32 for numerical stability.
        calc_dtype = torch.float32
        x_calc = hidden_states.to(calc_dtype)

        # Prepare a broadcastable mask (`mask_calc`).
        # If no mask is provided, treat all elements as valid
        # (mask_calc is all ones).
        # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
        mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)

        # Cumulative Statistics Calculation
        # 1. Sum of values over reduction axes at each time step.
        sum_values_at_t = torch.sum(x_calc,
                                    dim=self.reduction_axes,
                                    keepdim=True)
        # 2. Cumulative sum of values over time.
        cum_sum_values = torch.cumsum(sum_values_at_t, dim=1)

        # 3. Count of valid elements in the normalization group at each time step.
        #    (A "group" here consists of all features at a given Batch, Time).
        elements_in_group_at_t = torch.sum(mask_calc,
                                           dim=self.reduction_axes,
                                           keepdim=True)
        # 4. Cumulative count of valid elements over time.
        cum_count_elements = torch.cumsum(elements_in_group_at_t, dim=1)
        # Avoid division by zero if all preceding elements were masked.
        safe_cum_count_elements = torch.clamp(cum_count_elements, min=1.0)

        # 5. Cumulative mean.
        cum_mean = cum_sum_values / safe_cum_count_elements

        # 6. Sum of squared differences from the cumulative mean.
        #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
        #    Using x_calc here for the difference, as cum_mean already accounts for masking.
        squared_diff_from_mean = (x_calc - cum_mean).pow(2)
        sum_sq_diff_at_t = torch.sum(squared_diff_from_mean,
                                     dim=self.reduction_axes,
                                     keepdim=True)

        # 7. Cumulative sum of squared differences over time.
        cum_sum_sq_diff = torch.cumsum(sum_sq_diff_at_t, dim=1)

        # 8. Cumulative variance.
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements

        # Normalize the input using the calculated cumulative statistics:
        # (x - E[x]) / sqrt(Var[x] + eps)
        normalized_x = (x_calc - cum_mean) * torch.rsqrt(cum_variance +
                                                         self.eps)

        # Apply affine transformation (scale and bias) if enabled.
        # Scale and bias are applied per-channel (last dimension).
        scale = self.weight.to(calc_dtype)
        # Reshape for broadcasting: [C] -> [1, ..., 1, C]
        scale_view_shape = [1] * (hidden_states.dim() - 1) + [
            self.num_channels
        ]
        normalized_x = normalized_x * scale.view(scale_view_shape)

        # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
        # This ensures padded/invalid positions in the input result in zero output.
        final_output = normalized_x * mask_calc

        return final_output.to(input_dtype)


class ConformerNoAttEncoderLayer(torch.nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        feed_forward: Optional[torch.nn.Module] = None,
        feed_forward_macaron: Optional[torch.nn.Module] = None,
        conv_module: Optional[torch.nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = WENET_NORM_CLASSES[layer_norm_type](
            size, eps=norm_eps)  # for the FNN module
        self.norm_mha = WENET_NORM_CLASSES[layer_norm_type](
            size, eps=norm_eps)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)  # for the CNN module
            self.norm_final = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)  # for the final output of the block
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
                (0, 0, 0) means fake mask.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, new_cnn_cache


class ConformerNoPosEncoderLayer(ConformerEncoderLayer):
    """Encoder layer module.

    conv berfore attention

    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: T_CACHE = (torch.zeros(
            (0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class Conformer(torch.nn.Module):
    """
    https://arxiv.org/pdf/2204.06164
    https://ieeexplore.ieee.org/document/9747879
    """

    def __init__(self, config) -> None:
        super().__init__()

        activation = WENET_ACTIVATION_CLASSES[config.activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            config.attention_heads,
            config.output_size,
            config.attention_dropout_rate,
            config.query_bias,
            config.key_bias,
            config.value_bias,
            config.use_sdpa,
            config.n_kv_head,
            config.head_dim,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            config.output_size,
            config.linear_units,
            config.dropout_rate,
            activation,
            config.mlp_bias,
            config.n_expert,
            config.n_expert_activated,
        )
        # convolution module definition
        convolution_layer_args = (
            config.output_size,
            config.cnn_module_kernel,
            activation,
            config.cnn_module_norm,
            config.causal,
            config.conv_bias,
            config.conv_norm_eps,
            config.conv_inner_factor,
        )
        mlp_class = WENET_MLP_CLASSES[config.mlp_type]

        self.first_n_encoders = torch.nn.ModuleList([
            ConformerNoAttEncoderLayer(
                config.output_size,
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args)
                if config.macaron_style else None,
                ConvolutionModule(*convolution_layer_args)
                if config.use_cnn_module else None,
                config.dropout_rate,
                config.normalize_before,
                layer_norm_type=config.layer_norm_type,
                norm_eps=config.norm_eps,
            ) for _ in range(config.first_n_layers)
        ])

        self.causal_encoders = torch.nn.ModuleList([
            ConformerNoPosEncoderLayer(
                config.output_size,
                WENET_ATTENTION_CLASSES[config.selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args)
                if config.macaron_style else None,
                ConvolutionModule(*convolution_layer_args)
                if config.use_cnn_module else None,
                config.dropout_rate,
                config.normalize_before,
                layer_norm_type=config.layer_norm_type,
                norm_eps=config.norm_eps,
            ) for _ in range(config.causal_blocks)
        ])

        self.noncausal_encoders = torch.nn.ModuleList([
            ConformerNoPosEncoderLayer(
                config.output_size,
                WENET_ATTENTION_CLASSES[config.selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args)
                if config.macaron_style else None,
                ConvolutionModule(*convolution_layer_args)
                if config.use_cnn_module else None,
                config.dropout_rate,
                config.normalize_before,
                layer_norm_type=config.layer_norm_type,
                norm_eps=config.norm_eps,
            ) for _ in range(config.noncausal_blocks)
        ])

        self.after_norm = WENET_NORM_CLASSES[config.layer_norm_type](
            config.output_size, config.norm_eps)
        self.config = config

    def forward(self, xs: torch.Tensor, masks: torch.Tensor):
        """ Forward for training
        """
        masks = masks.unsqueeze(1)  # (B,1,T)
        causal_att_mask = causal_or_lookahead_mask(masks, 0,
                                                   self.config.left_context)
        noncausal_att_mask = causal_or_lookahead_mask(
            masks, self.config.right_context, self.config.left_context)
        mask_pad = masks
        if self.config.use_sdpa:
            noncausal_att_mask = mask_to_bias(noncausal_att_mask, xs.dtype)
            causal_att_mask = mask_to_bias(causal_att_mask, xs.dtype)
        if self.config.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, causal_att_mask,
                                                  noncausal_att_mask, mask_pad)
        else:
            xs = self.forward_layers(xs, causal_att_mask, noncausal_att_mask,
                                     mask_pad)
        if self.config.final_norm:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: torch.Tensor,
                       causal_chunk_masks: torch.Tensor,
                       noncausal_chunk_masks: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:

        for layer in self.first_n_encoders:
            xs, _ = layer(xs, mask_pad)
        for layer in self.causal_encoders:
            xs, _, _, _ = layer(xs, causal_chunk_masks, mask_pad)
        for layer in self.noncausal_encoders:
            xs, _, _, _ = layer(xs, noncausal_chunk_masks, mask_pad)
        return xs

    def forward_layers_checkpointed(self, xs: torch.Tensor,
                                    causal_chunk_masks: torch.Tensor,
                                    noncausal_chunk_masks: torch.Tensor,
                                    mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.first_n_encoders:
            xs, _ = ckpt.checkpoint(layer.__call__,
                                    xs,
                                    mask_pad,
                                    use_reentrant=False)
        for layer in self.causal_encoders:
            xs, _, _, _ = ckpt.checkpoint(layer.__call__,
                                          xs,
                                          causal_chunk_masks,
                                          mask_pad,
                                          use_reentrant=False)
        for layer in self.noncausal_encoders:
            xs, _, _, _ = ckpt.checkpoint(layer.__call__,
                                          xs,
                                          noncausal_chunk_masks,
                                          mask_pad,
                                          use_reentrant=False)

        return xs


if __name__ == '__main__':
    from efficient_conformer.configs.default import get_config

    config = get_config()
    config.gradient_checkpointing = False
    model = Conformer(config)

    print(model)

    xs = torch.rand(2, 100, 256)
    xs_lens = torch.tensor([10, 100])
    masks = ~make_pad_mask(xs_lens)

    out, mask = model(xs, masks)
    print(out, mask.sum(-1))
