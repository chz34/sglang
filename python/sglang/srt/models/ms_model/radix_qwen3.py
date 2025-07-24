import os
import time
import math
import logging
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple, Union, Type

import numpy as np

import torch
# from torch import nn

import mindspore as ms
from mindspore import nn
from mindspore import Tensor, JitConfig, Model, mutable, dtype, Parameter
from mindspore.communication.management import get_group_size
from mindspore.communication.comm_func import barrier
from mindspore.nn.utils import no_init_parameters
from mindspore import ops, mint, jit
from mindspore.ops.operations.nn_ops import IncreFlashAttention, FlashAttentionScore

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead


from .utils import tensor_pt2ms

logger = logging.getLogger(__name__)


class MsNativeAttnBackend(nn.Cell):
    """Paged Attention Manager."""
    def __init__(self,
                 config,
                 n_heads,
                 head_dim,
                 n_kv_heads,
                 seq_length=-1,
                 dtype=dtype.bfloat16):
        super().__init__()
        self.config = config
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.seq_length = seq_length
        self.is_first_iteration = True
        self.scale_value = 1 / math.sqrt(self.head_dim)
        self.key_cache = None
        self.value_cache = None
        self.dtype = dtype
        self.use_gpa = (self.n_heads != self.n_kv_heads)
        self.repeat_num = self.n_heads // self.n_kv_heads

        self.flash_attention = FlashAttentionScore(head_num=self.n_heads,
                                                   scale_value=self.scale_value,
                                                   next_tokens=0,
                                                   input_layout="TH")
        self.incre_flash_attention = IncreFlashAttention(num_heads=self.n_heads,
                                                        input_layout="BSH",
                                                        scale_value=self.scale_value,
                                                        num_key_value_heads=self.n_kv_heads)

    # pylint: disable=W0613
    def construct(self, key, value, key_cache=None, value_cache=None, out_cache_loc=None, k_scale=None, v_scale=None):
        if k_scale is not None:
            key = key / k_scale
        if v_scale is not None:
            value = value / v_scale
        key = mint.reshape(key, (-1, self.n_kv_heads, self.head_dim))
        value = mint.reshape(value, (-1, self.n_kv_heads, self.head_dim))

        key = ops.depend(key, ops.scatter_update(key_cache, out_cache_loc, key))
        key = ops.depend(key, ops.scatter_update(value_cache, out_cache_loc, value))

        return key

    def extend(self, query, key, value, attn_mask=None, alibi_mask=None, prefix=None, padding_mask=None,
               q_seq_lens=None, batch_valid_length=None):
        _, _, _, output = self.flash_attention(query,
                                               key,
                                               value,
                                               alibi_mask,
                                               None,
                                               padding_mask,
                                               attn_mask,
                                               prefix,
                                               q_seq_lens,
                                               batch_valid_length)
        return output

    def decode(self, query, batch_valid_length, attn_mask=None, q_seq_lens=None,
                    key_cache=None, value_cache=None, token_cache_loc=None, kv_mask=None):
        return self.decode_ifa(query, batch_valid_length, attn_mask, q_seq_lens,
                                key_cache, value_cache, token_cache_loc, kv_mask)

    def decode_ifa(self, query, batch_valid_length, attn_mask=None, q_seq_lens=None,
                    key_cache=None, value_cache=None, token_cache_loc=None, kv_mask=None):
        # B, S, H
        query = mint.reshape(query, (-1, 1, self.n_heads * self.head_dim))

        bs = batch_valid_length.shape[0]

        key = ops.gather(key_cache, token_cache_loc, 0)
        value = ops.gather(value_cache, token_cache_loc, 0)

        # B, S, H
        key = mint.reshape(key, (bs, -1, self.n_kv_heads * self.head_dim))
        value = mint.reshape(value, (bs, -1, self.n_kv_heads * self.head_dim))

        output = self.incre_flash_attention(
            query,
            [key],
            [value],
            None,
            batch_valid_length
        )

        output = mint.reshape(output, (-1, self.n_heads * self.head_dim))
        return output



class VocabEmbedding(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()

        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size

        self.gather = ops.Gather()

        self.weight = Parameter(mint.zeros(
            (self.num_embeddings, self.embedding_dim),
            dtype=config.param_dtype), requires_grad=False)

    def construct(self, input: Tensor) -> Tensor:
        return self.gather(self.weight, input, 0)


class RMSNorm(nn.Cell):
    def __init__(
            self, norm_dim: int, eps: float, param_dtype: Optional[Type]) -> None:
        super().__init__()
        
        self.weight = Parameter(mint.ones(norm_dim, dtype=param_dtype))
        self.rms_norm = ops.RmsNorm(eps)

    def construct(
            self,
            x: Tensor,
            residual: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x
        output = self.rms_norm(x, self.weight)[0]
        if residual is None:
            return output
        return output, residual

class Qwen3ColParallelLinear(nn.Cell):
    def __init__(self, input_size: int, output_size: int, param_dtype: Optional[Type], bias: bool) -> None:
        super().__init__()
        
        self.tp_size = 2

        self.param_dtype = param_dtype
        self.input_size = input_size
        self.output_size = output_size // self.tp_size
        self.enable_bias = bias

        self.matmul = ops.MatMul(transpose_b=True)
        self.weight = Parameter(
            mint.zeros(
                (self.output_size, self.input_size),
                dtype=self.param_dtype,
            ),
            requires_grad=False,
        )

        if self.enable_bias:
            self.bias_add = ops.Add()
            self.bias = Parameter(
                mint.zeros(self.output_size, dtype=self.param_dtype)
            )

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = self.bias_add(x, self.bias)
        return x.view(*origin_shape[:-1], -1)
    
    
class Qwen3RowParallelLinear(nn.Cell):
    def __init__(self, input_size: int, output_size: int, param_dtype: Optional[Type], bias: bool) -> None:
        super().__init__()
        
        self.tp_size = 2

        self.param_dtype = param_dtype
        self.input_size = input_size // self.tp_size
        self.output_size = output_size
        self.enable_bias = bias

        self.matmul = ops.MatMul(transpose_b=True)
        self.weight = Parameter(
            mint.zeros(
                (self.output_size, self.input_size),
                dtype=self.param_dtype,
            ),
            requires_grad=False,
        )

        if self.enable_bias:
            self.bias_add = ops.Add()
            self.bias = Parameter(
                mint.zeros(self.output_size, dtype=self.param_dtype)
            )

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = self.bias_add(x, self.bias)
            
        return x.view(*origin_shape[:-1], -1)


class Qwen3Linear(nn.Cell):
    def __init__(self, input_size: int, output_size: int, param_dtype: Optional[Type], bias: bool) -> None:
        super().__init__()

        self.param_dtype = param_dtype
        self.input_size = input_size
        self.output_size = output_size
        self.enable_bias = bias

        self.matmul = ops.MatMul(transpose_b=True)
        self.weight = Parameter(
            mint.zeros(
                (self.output_size, self.input_size),
                dtype=self.param_dtype,
            ),
            requires_grad=False,
        )

        if self.enable_bias:
            self.bias_add = ops.Add()
            self.bias = Parameter(
                mint.zeros(self.output_size, dtype=self.param_dtype)
            )

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = self.bias_add(x, self.bias)
        return x.view(*origin_shape[:-1], -1)


class Qwen3MLP(nn.Cell):
    def __init__(
            self,
            config
    ) -> None:
        super().__init__(config)
        
        self.up_proj = Qwen3Linear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dtype=config.param_dtype,
            bias=False
        )
        self.gate_proj = Qwen3Linear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dtype=config.param_dtype,
            bias=False
        )
        self.down_proj = Qwen3Linear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            param_dtype=config.param_dtype,
            bias=False
        )
        self.act_fn = ops.silu

    def construct(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float = 10000,
                              max_position_embeddings: int = 2048) -> float:
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(low: float, high: float, dim: int,
                           dtype: np.dtype) -> np.ndarray:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = np.clip(linear_func, 0, 1)
    return ramp_func


class InferYaRNScalingRotaryEmbedding(nn.Cell):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * attn_factor)
        
        super().__init__()
        
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.gather = ops.Gather()
        
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, scaling_factor: float) -> Tensor:
        pos_freqs = self.base**(
            np.arange(0, self.rotary_dim, 2, dtype=np.float32) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(
                low,
                high,
                self.rotary_dim // 2,
                dtype=np.float32  # type: ignore[arg-type]
            )) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.scaling_factor)
        t = np.arange(self.max_position_embeddings *
                      self.scaling_factor).astype(np.float32)
        self.freqs = Tensor(freqs.reshape(1, 1, 1, -1), dtype=self.dtype)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb) * self.mscale  # (seq_len, head_dim)
        freqs_sin = np.sin(emb) * self.mscale  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin
    
    def construct(
            self,
            positions: Tensor,
            query: Tensor,
            key: Tensor,
            batch_valid_length: Tensor,
            is_prefill: bool,
            offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()

        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = self.gather(self.freqs_cos, positions.view(-1), 0)
            freqs_sin = self.gather(self.freqs_sin, positions.view(-1), 0)

        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin, batch_valid_length)


class Qwen3RotaryEmbedding(nn.Cell):
    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            dtype: Optional[Type]
    ) -> None:
        super().__init__()
        
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype
        
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.gather = ops.Gather()
        
        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        freqs_base = mint.arange(0, self.rotary_dim,
                                 2).astype(np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (base ** (freqs_base / self.rotary_dim))  # (head_dim // 2, )
        return freqs

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.base)
        t = np.arange(0, self.max_position_embeddings, 1).astype(np.float32)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb)  # (seq_len, head_dim)
        freqs_sin = np.sin(emb)  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(
            self,
            positions: Tensor,
            query: Tensor,
            key: Tensor,
            batch_valid_length: Tensor,
            is_prefill: bool,
            offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()

        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = self.gather(self.freqs_cos, positions.view(-1), 0)
            freqs_sin = self.gather(self.freqs_sin, positions.view(-1), 0)

        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin, batch_valid_length)
    

class Qwen3Attention(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = float(self.head_dim ** -0.5)
        self.rope_theta = int(config.rope_theta)
        self.param_dtype = config.param_dtype
        self.max_position = config.max_position_embeddings
        self.rope_type = config.rope_scaling["rope_type"]
        self.rope_factor = config.rope_scaling["factor"]
        self.rope_max_position_embeddings = config.rope_scaling["original_max_position_embeddings"]
        
        self.attn = MsNativeAttnBackend(config,
                                    config.num_attention_heads,
                                    config.head_dim,
                                    config.num_key_value_heads,
                                    config.max_position_embeddings,
                                    dtype=config.param_dtype)        

        self.q_proj = Qwen3Linear(
            input_size=self.hidden_size,
            output_size=self.q_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias
        )
        self.q_norm = RMSNorm(norm_dim=config.head_dim,
                                    eps=config.rms_norm_eps, param_dtype=config.param_dtype)
        self.k_proj = Qwen3Linear(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias
        )
        self.k_norm = RMSNorm(norm_dim=config.head_dim,
                                    eps=config.rms_norm_eps, param_dtype=config.param_dtype)
        self.v_proj = Qwen3Linear(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias
        )
        self.o_proj = Qwen3Linear(
            input_size=self.q_size,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias
        )
        self.rotary_emb = None
        if self.rope_type == "yarn":
            self.rotary_emb = InferYaRNScalingRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.rope_max_position_embeddings,
                base=self.rope_theta,
                is_neox_style=True,
                scaling_factor=self.rope_factor,
                dtype=self.param_dtype,
            )            
        else:
            self.rotary_emb = Qwen3RotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.max_position,
                base=self.rope_theta,
                dtype=self.param_dtype,
            )

    def construct(self, hidden_state: Tensor, positions: Tensor, batch_valid_length: Tensor, is_prefill: bool, layer_idx: int, attn_mask: Tensor, q_seq_lens: Tensor, key_cache: Tensor, value_cache: Tensor, out_cache_loc: Tensor,
                  token_cache_loc: Tensor, kv_mask: Tensor)  -> Tensor:
        token_lens, hidden_dim = hidden_state.shape

        q = self.q_proj(hidden_state).view(-1, self.num_heads, self.head_dim).contiguous()
        k = self.k_proj(hidden_state).view(-1, self.num_kv_heads, self.head_dim).contiguous()
        v = self.v_proj(hidden_state).view(-1, self.kv_size).contiguous()

        q = self.q_norm(q).view(-1, self.q_size)
        k = self.k_norm(k).view(-1, self.kv_size)
        
        q, k = self.rotary_emb(positions, q, k,
                               batch_valid_length=batch_valid_length, is_prefill=is_prefill)

                    
        k = k.contiguous()
        v = v.contiguous()

        key_out = self.attn(k, v, key_cache=key_cache, value_cache=value_cache, out_cache_loc=out_cache_loc)
        q = ops.depend(q, key_out)

        if is_prefill:
            attn_output = self.attn.extend(q, k, v, attn_mask, None, None, None,
                                             q_seq_lens, batch_valid_length)
        else:
            attn_output = self.attn.decode(q, batch_valid_length, attn_mask, q_seq_lens, key_cache, value_cache,
                                             token_cache_loc, kv_mask)
    
        output = self.o_proj(attn_output).view(token_lens, -1)
        return output


class Qwen3DecoderLayer(nn.Cell):
    def __init__(
            self, config) -> None:
        super().__init__()
        
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config)
        self.mlp = Qwen3MLP(config=config)
        self.input_layernorm = RMSNorm(norm_dim=config.hidden_size, eps=config.rms_norm_eps, param_dtype=config.param_dtype)
        self.post_attention_layernorm = RMSNorm(norm_dim=config.hidden_size, eps=config.rms_norm_eps, param_dtype=config.param_dtype)

    def construct(self, hidden_state: Tensor, residual: Tensor, positions: Tensor, batch_valid_length: Tensor, is_prefill: bool, layer_idx: int, attn_mask: Tensor, q_seq_lens: Tensor, key_cache: Tensor, value_cache: Tensor, out_cache_loc: Tensor,
                  token_cache_loc: Tensor, kv_mask: Tensor) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
        else:
            hidden_state, residual = self.input_layernorm(hidden_state, residual)
        hidden_state = self.self_attn(
            hidden_state=hidden_state,
            positions=positions,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            layer_idx=layer_idx,
            attn_mask=attn_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
            token_cache_loc=token_cache_loc,
            kv_mask=kv_mask
        )
        hidden_state, residual = self.post_attention_layernorm(hidden_state, residual)
        hidden_state = self.mlp(hidden_state)

        return hidden_state, residual

class Qwen3Model(nn.Cell):
    r"""
    qwen3 model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabEmbedding(config=config)
        self.layers = nn.CellList()
        for i in range(self.num_hidden_layers):
            layer = Qwen3DecoderLayer(config=config)
            self.layers.append(layer)

        self.norm = RMSNorm(norm_dim=config.hidden_size, eps=config.rms_norm_eps, param_dtype=config.param_dtype)

    # def add_flags_custom(self, is_first_iteration):
    #     """Add customized attributes for specific cells in the model."""
    #     self.add_flags(is_first_iteration=is_first_iteration)
    #     self.model.add_flags(is_first_iteration=is_first_iteration)
    #     for layer in self.model.layers:
    #         layer.add_flags(is_first_iteration=is_first_iteration)
    #         layer.attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    @jit(jit_level="O0", infer_boost="on")
    def construct(self, input_ids,  position_ids=None, attention_mask=None,
                  batch_valid_length=None, batch_index=None, zactivate_len=None,
                  prefix_keys_values=None, is_prefill=True,
                  q_seq_lens=None, key_cache=None, value_cache=None, out_cache_loc=None,
                  token_cache_loc=None, kv_mask=None):
        """
        Forward of qwen model.
        """
        hidden_state = self.embed_tokens(input_ids)
        residual = None
        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            hidden_state, residual = layer(
                hidden_state=hidden_state,
                residual=residual,
                positions=position_ids,
                batch_valid_length=batch_valid_length,
                is_prefill=is_prefill,
                layer_idx=i,
                attn_mask=attention_mask,
                q_seq_lens=q_seq_lens,
                key_cache=key_cache[i],
                value_cache=value_cache[i],
                out_cache_loc=out_cache_loc,
                token_cache_loc=token_cache_loc,
                kv_mask=kv_mask
            )

        hidden_state, _ = self.norm(hidden_state, residual)

        return hidden_state
    

class Qwen3ForCausalLM(nn.Cell):
    def __init__(self,
                model_config: ModelConfig,
                load_config: LoadConfig,
                prefix: str = "",) -> None:
        super().__init__()
        
        ms.set_context(infer_boost="on")
        ms.set_context(mode=ms.context.PYNATIVE_MODE)
        ms.set_context(graph_kernel_flags="--disable_pass=gather_pre_rms_norm_fusion")
        
        self.prev_prefill = False

        self.model_config = model_config
        self.config = model_config.hf_config
        setattr(self.config, "param_dtype", dtype.bfloat16)
        pynative_model = Qwen3Model(self.config)
        self.model = pynative_model
        
        self.lm_head = Qwen3Linear(
            input_size=self.config.hidden_size,
            output_size=self.config.vocab_size,
            param_dtype=self.config.param_dtype,
            bias=False
        )
        self.gather = ops.Gather()
        
        
        self.lower_triangle_mask = Tensor(
            np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1), dtype=self.config.param_dtype
        )
        self.key_cache = []
        self.value_cache = []
    
    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=dtype.int32)
        num_kv_heads = self.config.num_key_value_heads
        head_size = self.config.head_dim
        kv_cache_shape = (None, num_kv_heads, head_size)
        kv_cache_dtype = self.config.param_dtype
        num_layers = self.config.num_hidden_layers

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])

        dyn_out_cache_loc = Tensor(shape=[None, ], dtype=dtype.int32)
        dyn_attention_mask = Tensor(shape=[None, None], dtype=self.config.param_dtype)
        dyn_kv_mask = Tensor(shape=[None, None, None, None], dtype=dtype.bool_)
        dyn_batch_valid_length = Tensor(shape=[None, ], dtype=dtype.int32)
        dyn_q_seq_lens = Tensor(shape=[None, ], dtype=dtype.int32)
        dyn_token_cache_loc = Tensor(shape=[None, None], dtype=dtype.int32)

        self.model.set_inputs(
            input_ids=dyn_input_ids,
            position_ids=dyn_position_ids,
            attention_mask=dyn_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            value_cache=dyn_value_caches,
            out_cache_loc=dyn_out_cache_loc,
            token_cache_loc=dyn_token_cache_loc,
            kv_mask=dyn_kv_mask
        )


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        param_dict = self.parameters_dict()
        
        for (name, weight) in weights:
            if name in param_dict:
                param = param_dict[name]
                param.set_data(tensor_pt2ms(weight))
                
    def get_kvcache(self, forward_batch: ForwardBatch):
        if self.key_cache and self.value_cache:
            return mutable(self.key_cache), mutable(self.value_cache)

        for i in range(self.config.num_hidden_layers):
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(i)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(i)
            self.key_cache.append(mint.zeros(list(k_cache.shape), dtype=ms.bfloat16))
            self.value_cache.append(mint.zeros(list(v_cache.shape), dtype=ms.bfloat16))

        return mutable(self.key_cache), mutable(self.value_cache)
    
    def prepare_token_cache_loc_with_mask(self, forward_batch: ForwardBatch):
        """ prepare the token cache loc and mask """
        batch_size = forward_batch.batch_size
        max_len = forward_batch.seq_lens.max().item()
        token_cache_loc = torch.zeros([batch_size, max_len], dtype=torch.int32)
        kv_mask = torch.ones([batch_size, 1, 1, max_len], dtype=torch.bool)
        for i in range(batch_size):
            cur_seq_length = forward_batch.seq_lens[i]
            req_pool_idx = forward_batch.req_pool_indices[i]
            per_req_tokens = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :cur_seq_length]
            token_cache_loc[i, :cur_seq_length] = per_req_tokens
            kv_mask[i, 0, 0, :cur_seq_length] = False

        return tensor_pt2ms(token_cache_loc), tensor_pt2ms(kv_mask)
        
    def prepare_inputs(self, input_ids, positions, forward_batch):
        key_cache, value_cache = self.get_kvcache(forward_batch)

        is_prefill = forward_batch.forward_mode.is_extend()

        query_lens_np = forward_batch.seq_lens.cpu().numpy()
        batch_valid_length = forward_batch.seq_lens.cpu().numpy()

        # always not finished in the model forward
        is_finished = [False for _ in range(forward_batch.batch_size)]

        if is_prefill:
            q_seq_lens = query_lens_np
        else:
            q_seq_lens = np.ones([forward_batch.batch_size], dtype=np.int32)
        
        token_cache_loc, kv_mask = self.prepare_token_cache_loc_with_mask(forward_batch)

        model_inputs = {}
        model_inputs["input_ids"] = tensor_pt2ms(input_ids).to(ms.int32)
        model_inputs["batch_valid_length"] = ms.Tensor(batch_valid_length, dtype=ms.int32)
        model_inputs["position_ids"] = tensor_pt2ms(positions).to(ms.int32)
        model_inputs["q_seq_lens"] = ms.Tensor(q_seq_lens, dtype=ms.int32)
        model_inputs["attention_mask"] = self.lower_triangle_mask
        model_inputs["out_cache_loc"] = ms.Tensor(
            forward_batch.out_cache_loc.cpu().numpy(), dtype=ms.int32
        )
        model_inputs["token_cache_loc"] = token_cache_loc
        model_inputs["kv_mask"] = kv_mask
        model_inputs["is_prefill"] = is_prefill
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache
        
        return model_inputs

    def construct(self):
        return

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                forward_batch: ForwardBatch) -> Tensor:

        model_inputs = self.prepare_inputs(input_ids, positions, forward_batch)
        batch_valid_length = model_inputs["batch_valid_length"]
        is_prefill = model_inputs["is_prefill"]

        if self.prev_prefill != is_prefill:
            self.set_model_inputs(is_prefill)
        self.prev_prefill = is_prefill

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        pre_gather = is_prefill and batch_valid_length is not None
        start_time = time.time()
        hidden_state = self.model(**model_inputs)
        end_time = time.time()
        logger.info(f"Phase: {self.model.phase}, run model time: {end_time - start_time}s")
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            hidden_state = self.gather(hidden_state, batch_valid_length - 1, 0)
        logits = self.lm_head(hidden_state)
        logits = ops.cast(logits, dtype.float32)
        logits = ops.reshape(logits, (-1, logits.shape[-1]))
        logits_result = LogitsProcessorOutput(
            next_token_logits=torch.Tensor(logits.asnumpy()).to(input_ids.device)
        )
        return logits_result

