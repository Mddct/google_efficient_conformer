import math
from typing import Any, Optional, Sequence, Tuple

import torch

T_CACHE = Tuple[torch.Tensor, torch.Tensor]
Config = Any


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


# copy from:https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L84
#
def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# modified from:
#     https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L95
def google_apply_rotary_emb(x: torch.Tensor,
                            freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    return x_out


class RMSNorm(torch.nn.Module):
    """ https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.add_unit_offset = add_unit_offset

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return x * (1 + self.weight)
        else:
            return x * self.weight


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer.
    if n_kv_head != None and n_kv_head != n_head
    see: https://arxiv.org/pdf/1911.02150.pdf
         https://arxiv.org/pdf/2305.13245.pdf

    Example:
        case 1: n_kv_head == None, head_dim == None, MultiHead attention (MHSA)
        case 2: n_kv_head=1, n_head = 16, MultiQuery attention (MQA)
        case 3: nv_kv_head=2, n_head = 16, GroupedQuery attention (GQA)

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.inner_dim = n_feat if head_dim is None else head_dim * n_head
        if n_kv_head is not None:
            assert head_dim is not None
            self.inner_kv_dim = head_dim * n_kv_head
            n_kv_head = n_kv_head
        else:
            self.inner_kv_dim = self.inner_dim
            n_kv_head = n_head
        # We assume d_v always equals d_k
        self.d_k = self.inner_dim // n_head
        assert self.d_k == self.inner_kv_dim // n_kv_head
        self.h = n_head
        self.h_kv = n_kv_head

        self.linear_q = torch.nn.Linear(n_feat,
                                        self.inner_dim,
                                        bias=query_bias)
        self.linear_k = torch.nn.Linear(n_feat,
                                        self.inner_kv_dim,
                                        bias=key_bias)
        self.linear_v = torch.nn.Linear(n_feat,
                                        self.inner_kv_dim,
                                        bias=value_bias)
        self.linear_out = torch.nn.Linear(self.inner_dim,
                                          n_feat,
                                          bias=query_bias)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.use_sdpa = use_sdpa
        self.dropout_rate = dropout_rate

    def _forward_linearx(self,
                         name: str,
                         x: torch.Tensor,
                         head_first: bool = True) -> torch.Tensor:
        assert x.ndim >= 3
        if name == 'query':
            x = self.linear_q(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h, self.d_k])
        elif name == 'key':
            x = self.linear_k(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])
        else:
            assert name == 'value'
            x = self.linear_v(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])

        # split last dim
        x = x.view(x_shape)
        if head_first:
            x = x.transpose(-3,
                            -2)  # (batch, ...,  head or head_kv, time, d_k)
        return x

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, ..., time1, size).
            key (torch.Tensor): Key tensor (#batch, ..., time2, size).
            value (torch.Tensor): Value tensor (#batch, ..., time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, ..., n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, ..., n_head_kv, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, ..., n_head_kv, time2, d_k).

        """
        q = self._forward_linearx('query', query)
        k = self._forward_linearx('key', key)
        v = self._forward_linearx('value', value)
        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, ..., n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, ..., n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, ..., time1, time2), (0, ..., 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(-1) > 0:  # time2 > 0
            mask = mask.unsqueeze(-3).eq(0)  # (batch, .., 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[..., :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores.float(),
                                 dim=-1).type_as(value).masked_fill(
                                     mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores.float(), dim=-1).type_as(
                value)  # (batch, ..., head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, ...,  head, time1, d_k)
        x = x.transpose(-3, -2).contiguous()  # [batch, ..., time1, head, d_k]
        x_shape = x.size()[:-2] + torch.Size([self.h * self.d_k])
        x = x.view(x_shape)  # (batch, ..., time1, d_model)
        return self.linear_out(x)  # (batch, ...,  time1, d_model)

    def _update_kv_and_cache(
            self,
            k: torch.Tensor,
            v: torch.Tensor,
            cache: T_CACHE,
            head_first: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE]:
        new_cache = cache
        seq_axis = -2 if head_first else -3
        head_axis = -3 if head_first else -2
        if not self.training:
            # NOTE(xcsong):
            #   when export onnx model, for 1st chunk, we feed
            #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
            #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
            #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
            #       and we will always do splitting and
            #       concatnation(this will simplify onnx export). Note that
            #       it's OK to concat & split zero-shaped tensors(see code below).
            #   when export jit  model, for 1st chunk, we always feed
            #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
            # >>> a = torch.ones((1, 2, 0, 4))
            # >>> b = torch.ones((1, 2, 3, 4))
            # >>> c = torch.cat((a, b), dim=2)
            # >>> torch.equal(b, c)        # True
            # >>> d = torch.split(a, 2, dim=-1)
            # >>> torch.equal(d[0], d[1])  # True
            key_cache, value_cache = cache
            if key_cache.size(0) > 0:
                k = torch.cat([key_cache, k], dim=seq_axis)
            if value_cache.size(0) > 0:
                v = torch.cat([value_cache, v], dim=seq_axis)
            # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
            #   non-trivial to calculate `next_cache_start` here.
            # new_cache = torch.cat((k, v), dim=-1) if not self.training else cache
            new_cache = (k, v)
        # for multi query or multi group attention
        if self.h_kv != self.h and self.h_kv != 1:
            # NOTE: onnxruntime issues:
            #     https://github.com/wenet-e2e/wenet/issues/2517
            # k = torch.repeat_interleave(
            #     k,
            #     self.h // self.h_kv,
            #     dim=-3,
            # )
            # v = torch.repeat_interleave(
            #     v,
            #     self.h // self.h_kv,
            #     dim=-3,
            # )
            n_repeat = self.h // self.h_kv
            k_shape = k.size()
            repeat_axis = head_axis + 1
            k = k.unsqueeze(head_axis).expand(
                k_shape[:repeat_axis] + torch.Size([n_repeat]) +
                k_shape[repeat_axis:]).reshape(
                    k_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) +
                    k_shape[repeat_axis:])
            v_shape = v.size()
            v = v.unsqueeze(head_axis).expand(
                v_shape[:repeat_axis] + torch.Size([n_repeat]) +
                v_shape[(repeat_axis):]).reshape(
                    v_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) +
                    v_shape[repeat_axis:])

        return k, v, new_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.unsqueeze(1),
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            output = (output.transpose(1, 2).contiguous().view(
                query.size(0), -1,
                self.h * self.d_k))  # (batch, time1, d_model)
            return self.linear_out(output), new_cache


class RopeMultiHeadedAttention(MultiHeadedAttention):

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa, n_kv_head, head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute rope scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q = self._forward_linearx('query', query, head_first=False)
        k = self._forward_linearx('key', key, head_first=False)
        v = self._forward_linearx('value', value, head_first=False)
        # NOTE(Mddct): In order to make the code easier to read,
        #    these two lines are not placed in MultiHeadedAttention.
        q = google_apply_rotary_emb(q, pos_emb)
        k = google_apply_rotary_emb(k, pos_emb)

        k, v, new_cache = self._update_kv_and_cache(k,
                                                    v,
                                                    cache,
                                                    head_first=False)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.unsqueeze(1),
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            output = (output.transpose(1, 2).contiguous().view(
                query.size(0), -1,
                self.h * self.d_k))  # (batch, time1, d_model)
            return self.linear_out(output), new_cache


class AudioConformerAttention(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        self.register_buffer("gradient_clipping",
                             torch.tensor(self.config.gradient_clipping),
                             persistent=False)
        self.post_in_features = self.config.hidden_size
        self.pre_attn_norm = RMSNorm(self.config.hidden_size)
        self.attn = RopeMultiHeadedAttention(
            config.n_head,
            config.hidden_size,
            config.dropout_rate,
            config.query_bias,
            config.key_bias,
            config.value_bias,
            config.use_sdpa,
            config.n_kv_head,
            head_dim=config.head_dim,
        )
        self.post = torch.nn.Linear(self.post_in_features,
                                    self.config.hidden_size,
                                    bias=False)
        self.post_norm = RMSNorm(self.config.hidden_size)

    def forward(self, xs: torch.Tensor, xs_mask: torch.BoolTensor,
                pos_emb: torch.Tensor) -> torch.Tensor:
        audio_encodings_input_to_attn = xs
        audio_encodings = torch.clamp(xs, -self.gradient_clipping,
                                      self.gradient_clipping)
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)
        # Output of self.attn is [B, T, D]
        audio_encodings_attn_out, _ = self.attn(audio_encodings_norm,
                                                audio_encodings_norm,
                                                audio_encodings_norm, xs_mask,
                                                pos_emb)
        audio_encodings = self.post(audio_encodings_attn_out)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping,
                                      self.gradient_clipping)
        return audio_encodings_input_to_attn + self.post_norm(audio_encodings)


class MLP(torch.nn.Module):
    """ https://arxiv.org/pdf/2002.05202.pdf
    """

    def __init__(
        self,
        config: Config,
    ):
        super().__init__()

        self.config = config
        self.register_buffer("gradient_clipping",
                             torch.tensor(self.config.gradient_clipping),
                             persistent=False)
        self.pre_layer_norm = RMSNorm(self.config.hidden_size)

        self.gate = torch.nn.Linear(config.hidden_size,
                                    4 * config.hidden_size,
                                    bias=False)
        # w_1 as up proj
        self.w_1 = torch.nn.Linear(config.hidden_size,
                                   4 * config.hidden_size,
                                   bias=False)
        # w_2 as down proj
        self.w_2 = torch.nn.Linear(4 * config.hidden_size,
                                   config.hidden_size,
                                   bias=False)

        self.post_layer_norm = RMSNorm(self.config.hidden_size)
        self.post_layer_scale = self.config.conf_residual_weight

    def forward(self, x) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        residual = x
        x = torch.clamp(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_layer_norm(x)
        gate = torch.nn.functional.gelu(self.gate(x))
        up = self.w_1(x)
        fuse = gate * up
        out = self.w_2(fuse)
        out = torch.clamp(out, -self.gradient_clipping, self.gradient_clipping)

        out = self.post_layer_norm(out)
        return residual + self.post_layer_scale * out


class AudioLConv1d(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.pre_layer_norm = RMSNorm(self.config.hidden_size,
                                      eps=self.config.rms_norm_eps)
        self.linear_start = torch.nn.Linear(self.config.hidden_size,
                                            self.config.hidden_size * 2,
                                            bias=False)
        self.depthwise_conv1d = torch.nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=self.config.hidden_size,  # Depthwise
            bias=False,
        )
        self.register_buffer("gradient_clipping",
                             torch.tensor(self.config.gradient_clipping),
                             persistent=False)
        self.conv_norm = RMSNorm(self.config.hidden_size,
                                 eps=self.config.rms_norm_eps)
        self.linear_end = torch.nn.Linear(self.config.hidden_size,
                                          self.config.hidden_size,
                                          bias=False)

        self.causal_padding = self.config.conf_conv_kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # Save for residual connection

        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        audio_encodings = torch.nn.functional.glu(x, dim=-1)
        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)
        # Apply manual causal padding
        audio_encodings_permuted_padded = torch.nn.functional.pad(
            audio_encodings_permuted, (self.causal_padding, 0), 'constant',
            0.0)
        audio_encodings = self.depthwise_conv1d(
            audio_encodings_permuted_padded)
        # Permute back: [B, D, T_out] -> [B, T_out, D]
        audio_encodings = audio_encodings.permute(0, 2, 1)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping,
                                      self.gradient_clipping)
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = torch.nn.functional.gelu(audio_encodings)
        audio_encodings = self.linear_end(audio_encodings)
        output = audio_encodings + residual
        return output


class AudioConformerBlock(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.ffw_layer_start = MLP(self.config)
        self.attention = AudioConformerAttention(self.config)
        self.lconv1d = AudioLConv1d(self.config)
        self.ffw_layer_end = MLP(self.config)
        self.register_buffer("gradient_clipping",
                             torch.tensor(self.config.gradient_clipping),
                             persistent=False)
        self.norm = RMSNorm(self.config.hidden_size)

    def forward(self, x: torch.Tensor, x_att_mask: torch.BoolTensor,
                x_mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        audio_encodings = self.ffw_layer_start(x)
        audio_encodings = self.attention(audio_encodings, x_att_mask, pos_emb)
        validity_mask_for_lconv = x_mask[:, :, None]  # True for valid
        audio_encodings_for_lconv_input = audio_encodings * validity_mask_for_lconv.to(
            audio_encodings.dtype)
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping,
                                      self.gradient_clipping)
        output = self.norm(audio_encodings)
        return output


class AudioCumulativeGroupNorm(torch.nn.Module):
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

    def forward(self,
                hidden_states: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies cumulative group norm, optionally using a mask.

        Args:
          hidden_states: Input tensor, shape [B, T, *feature_dims, C].
          mask:  shape [B, T, *1, 1].

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

        if mask is None:
            # Prepare a broadcastable mask (`mask_calc`).
            # If no mask is provided, treat all elements as valid
            # (mask_calc is all ones).
            # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
            mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)
        else:
            mask_calc = mask.repeat(1, 1, x_calc.shape[-2], x_calc.shape[-1])

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


class AudioSSCPConvBlock(torch.nn.Module):
    """A single convolution block for the SubSampleConvProjection.

    This block consists of a 2D convolution, followed by CumulativeGroupNorm,
    and a ReLU activation. It handles manual padding for the convolution.
    """

    def __init__(
        self,
        config: Config,
        idx: int,
        input_freq_dim: int,  # Changed from input_spatial_dim
        manual_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        super().__init__()
        self.config = config
        self.manual_padding = manual_padding

        # in_channels is 1 for the first block, or C_out from previous block's conv
        in_channels = 1 if idx == 0 else self.config.sscp_conv_channel_size[idx
                                                                            -
                                                                            1]
        out_channels = self.config.sscp_conv_channel_size[idx]
        kernel_h, kernel_w = self.config.sscp_conv_kernel_size[idx]
        stride_h, stride_w = self.config.sscp_conv_stride_size[idx]

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(
                kernel_h,
                kernel_w,
            ),  # Kernel (kH, kW) operates on (Time, Freq_dim)
            stride=(stride_h, stride_w),
            padding=(0, 0),  # Manual padding is used
            bias=False,
        )

        # Calculate output frequency dimension (f_out_conv) after this convolution.
        # input_freq_dim is the unpadded width (feature dimension).
        # self.manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        f_in_padded = input_freq_dim + self.manual_padding[
            0] + self.manual_padding[1]
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1

        self.norm = AudioCumulativeGroupNorm(
            num_channels=out_channels,  # Channels of the conv output
            feature_dims=(
                f_out_conv, ),  # The frequency dimension size after conv
            eps=self.config.sscp_conv_group_norm_eps,
        )

        self.activation = torch.nn.ReLU()

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # Input audio_encodings is [B, C_in, T_in, F_in] (e.g., C_in=1)
        # manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        # F.pad applies to last two dims: F_in then T_in
        audio_encodings_padded = torch.nn.functional.pad(audio_encodings,
                                                         self.manual_padding,
                                                         mode="constant",
                                                         value=0.0)
        # Expected padded shape for F_in, k_w=3, pad_F=(1,1) -> F_padded = F_in+2
        # Expected padded shape for T_in, k_h=3, pad_T=(0,2) -> T_padded = T_in+2
        audio_encodings_conv = self.conv(audio_encodings_padded)

        # Expected conv output shape: [B, C_out, T_out, F_out]
        # Input to norm is [B, T_out, F_out, C_out]
        x_for_norm = audio_encodings_conv.permute(0, 2, 3, 1).contiguous()
        x_normed = self.norm(x_for_norm)
        # Output of norm is [B, T_out, F_out, C_out], permute back to [B, C_out, T_out, F_out]
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        return self.activation(audio_encodings_normed)


class AudioSubSampleConvProjection(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        current_f_for_block_input = config.input_feat_size  # Start with original feature dim
        calculated_block_padding = []
        calculated_f_out_dims = []  # Tracking frequency dimension output sizes

        for i in range(
                2):  # Assuming 2 conv layers as per sscp_conv_... arrays
            kernel_h, kernel_w = config.sscp_conv_kernel_size[i]
            stride_h, stride_w = config.sscp_conv_stride_size[i]

            pad_t_top = 0
            pad_t_bottom = kernel_h - 1

            pad_f_left = 1
            pad_f_right = 1

            manual_padding_tuple = (
                pad_f_left,
                pad_f_right,
                pad_t_top,
                pad_t_bottom,
            )
            calculated_block_padding.append(manual_padding_tuple)

            # Calculate output frequency dimension after this convolution
            # This uses the actual padding applied and kernel/stride.
            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (f_in_padded - kernel_w
                                ) // stride_w + 1  # Assuming dilation_w = 1
            calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv

        self.conv_0 = AudioSSCPConvBlock(
            idx=0,
            input_freq_dim=config.input_feat_size,  # Pass original feature dim
            config=config,
            manual_padding=calculated_block_padding[0],
        )
        self.conv_1 = AudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[
                0],  # Output freq dim from conv_0
            config=config,
            manual_padding=calculated_block_padding[1],
        )
        final_c_out = config.sscp_conv_channel_size[-1]
        final_f_out = calculated_f_out_dims[-1]  # Final frequency dimension
        self.input_proj_in_features = final_c_out * final_f_out
        self.input_proj_linear = torch.nn.Linear(self.input_proj_in_features,
                                                 self.config.hidden_size,
                                                 bias=False)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # audio_encodings is [B, T, F_in]
        # Reshape to [B, 1, T, F_in] (Batch, Channels=1, Height=Time, Width=F_in)
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)
        x = self.conv_0(audio_encodings_reshaped)
        x = self.conv_1(x)
        # x from conv_1 is [B, C_out_1, T_out_1, F_out_1]
        b, c_out, t_out, f_out = x.shape
        # Permute to [B, T_out_1, F_out_1, C_out_1] then flatten F_out_1 and C_out_1
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        output_flattened = x_permuted.view(b, t_out, f_out * c_out)
        output = self.input_proj_linear(output_flattened)
        return output


class AudioConformer(torch.nn.Module):
    """An audio encoder based on the [Universal Speech Model](https://arxiv.org/abs/2303.01037) architecture."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = AudioSubSampleConvProjection(config)
        self.conformer = torch.nn.ModuleList([
            AudioConformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        pe = precompute_freqs_cis(config.head_dim, config.rope_max_lens,
                                  config.rope_theta)
        self.register_buffer('pe', pe)

    def forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Encodes a batch of MELs (training only).

        Args:
            audio_mel: a torch.Tensor of shape [batch, num_frames, num_channels,
              mel_bins].

        Returns:
            audio_encodings: a torch.Tensor of shape
                `[batch_size, self.config.audio_soft_tokens_per_image,
                self.config.audio_config.hidden_size]`
            audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames].
        """
        audio_encodings = self.subsample_conv_projection(
            audio_mel)  # audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        t_sub = audio_encodings.shape[1]

        time_stride_product = 1
        for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):
            time_stride_product *= self.config.sscp_conv_stride_size[
                stride_pair_idx][0]

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        indices = torch.arange(
            t_sub, device=audio_mel_mask.device) * time_stride_product
        indices = torch.clamp(indices, max=audio_mel_mask.shape[1] -
                              1)  # Ensure indices are valid

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(audio_mel_mask.shape[0],
                                                  -1)  # [B, T_sub]
        elif (audio_mel_mask.ndim == indices.ndim
              and audio_mel_mask.shape[0] == 1 and indices.shape[0] != 1
              and t_sub == indices.shape[0]):
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            indices = indices.unsqueeze(0)

        current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub]
        pe = self.pe[None, :current_mask.shape[1],
                     None, :]  # [1, seq, q, head_dim // 2]
        # TODO: streaming mask
        att_mask = current_mask[:, :, None]
        if config.use_sdpa:
            att_mask = mask_to_bias(att_mask, audio_encodings.dtype)

        audio_encodings: torch.Tensor
        for block in self.conformer:
            audio_encodings = block(audio_encodings, att_mask, current_mask,
                                    pe)  # Pass the processed mask

        if self.config.conf_reduction_factor > 1:
            audio_encodings = audio_encodings[:, ::self.config.
                                              conf_reduction_factor]
            # Reduce the mask as well
            current_mask = current_mask[:, ::self.config.conf_reduction_factor]

        audio_encodings = audio_encodings * current_mask[:, :, None]
        return audio_encodings, current_mask


if __name__ == '__main__':
    from configs.default import get_config
    config = get_config()

    model = AudioConformer(config)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {:,d}'.format(num_params))

    input = torch.randn(2, 100, config.input_feat_size)
    mask = torch.ones(2, 100, dtype=torch.bool)

    out, out_mask = model(input, mask)
    print(out)
    print(out_mask.shape, out.shape)
