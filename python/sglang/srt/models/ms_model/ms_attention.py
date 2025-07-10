# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Paged Attention Manager for inference."""
import math

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, mint, ops
from mindspore.ops.operations.nn_ops import IncreFlashAttention, FlashAttentionScore

from mindformers.parallel_core.inference.utils import get_attn_mask_func
from mindformers.experimental.parallel_core.pynative.transformer.scale_mask_softmax import ScaleMaskSoftmax
from mindformers.experimental.parallel_core.pynative.utils import divide 

class CoreAttention(nn.Cell):
    r"""
    Get the weighted score along the seq_length.

    Args:
        config (dict): Configuration.

    Inputs:
        - **query** (Tensor) - Tensor of query matrix.
        - **key** (Tensor) - Tensor of key matrix.
        - **value** (Tensor) - Tensor of value matrix.
        - **attention_mask** (Tensor) - Tensor of attention mask matrix.

    Outputs:
        - **attn_output** (Tensor) - Tensor of shape :math:`(B, N, S, D)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config):
        super(CoreAttention, self).__init__()
        self.config = config
        self.compute_dtype = self.config.compute_dtype
        self.softmax_compute_dtype = self.config.softmax_compute_dtype
        self.sequence_parallel = self.config.parallel_config.use_sequence_parallel
        self.num_heads = self.config.num_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = divide(self.hidden_size, self.num_heads)

        coeff = None
        norm_factor = math.sqrt(self.head_dim)
        self.inv_norm_factor = Tensor(1.0 / norm_factor, dtype=self.compute_dtype)

        self.mask_func = get_attn_mask_func(self.config.mask_func_type)
        self.scale_mask_softmax = ScaleMaskSoftmax(self.mask_func,
                                                   softmax_compute_type=self.softmax_compute_dtype)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """
        Computes the attention scores, applies the attention mask, and returns the weighted
        sum of the value layer based on the attention probabilities.

        Inputs:
        ----------
        query_layer : Tensor
            The query tensor of shape [B, N, S_q, D].
        key_layer : Tensor
            The key tensor of shape [B, N, S_k, D].
        value_layer : Tensor
            The value tensor of shape [B, N, S_k, D].
        attention_mask : Tensor
            The attention mask tensor of shape [B, N, S_q, S_k].

        Returns:
        -------
        Tensor
            The attention output tensor of shape [B, N, S_q, D].
        """
        # score: [B, N, S_q, S_k]
        score = ops.bmm(query_layer, key_layer.transpose(-1, -2))
        score = mint.mul(score, self.inv_norm_factor)

        # attention scores and attention mask [B, N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        # [B, N, S_q, S_k] * [B, N, S_v, D] -> [B, N, S_q, D]
        weighted_values = ops.bmm(attention_probs, value_layer)

        return weighted_values


class MsAttnBackend(nn.Cell):
    """Paged Attention Manager."""
    def __init__(self,
                 config,
                 n_heads,
                 head_dim,
                 n_kv_heads,
                 kv_shape,
                 seq_length=-1,
                 dtype=mstype.bfloat16):
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
        self.repeat_num = divide(self.n_heads, self.n_kv_heads)

        self.flash_attention = FlashAttentionScore(head_num=self.n_heads,
                                                   scale_value=self.scale_value,
                                                   next_tokens=0,
                                                   input_layout="TH")
        self.incre_flash_attention = IncreFlashAttention(num_heads=self.n_heads,
                                                        input_layout="BSH",
                                                        scale_value=self.scale_value,
                                                        num_key_value_heads=self.n_kv_heads)

        self.core_attention = CoreAttention(self.config)

    # pylint: disable=W0613
    def construct(self, key, value, key_cache=None, value_cache=None, out_cache_loc=None, k_scale=None, v_scale=None):
        if k_scale is not None:
            key = key / k_scale
        if v_scale is not None:
            value = value / v_scale
        key = mint.reshape(key, (-1, self.n_kv_heads, self.head_dim))
        value = mint.reshape(value, (-1, self.n_kv_heads, self.head_dim))

        # key_cache[loc] = key.to(self.dtype)
        key = ops.depend(key, ops.scatter_update(key_cache, out_cache_loc, key))
        # value_cache[loc] = value.to(self.dtype)
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
        if self.incre_flash_attention is not None:
            return self.decode_ifa(query, batch_valid_length, attn_mask, q_seq_lens,
                                   key_cache, value_cache, token_cache_loc, kv_mask)
        else:
            return self.decode_core(query, batch_valid_length, attn_mask, q_seq_lens,
                                   key_cache, value_cache, token_cache_loc, kv_mask)

    def decode_core(self, query, batch_valid_length, attn_mask=None, q_seq_lens=None,
                    key_cache=None, value_cache=None, token_cache_loc=None, kv_mask=None):
        query = mint.reshape(query, (-1, 1, self.n_heads, self.head_dim))
        query = mint.transpose(query, 1, 2)

        bs = batch_valid_length.shape[0]

        req_key = key_cache[token_cache_loc]
        req_value = value_cache[token_cache_loc]

        # k: B,S,N,D -> B,N,D,S
        req_key = mint.transpose(req_key, 1, 2)
        # req_key = mint.transpose(req_key, 2, 3)

        # v: B,S,N,D -> B,N,S,D
        req_value = mint.transpose(req_value, 1, 2)

        if self.use_gpa:
            req_key = mint.repeat_interleave(req_key, repeats=self.repeat_num, dim=1)
            req_value = mint.repeat_interleave(req_value, repeats=self.repeat_num, dim=1)

        # q: B, N, S, D
        # k: B, N, D, S
        # v: B, N, S, D
        # o: B, N, S, D
        output = self.core_attention(
            query,
            req_key,
            req_value,
            kv_mask
        )

        output = mint.transpose(output, 1, 2)
        output = mint.reshape(output, (-1, self.n_heads * self.head_dim))

        return output
    
    def decode_ifa(self, query, batch_valid_length, attn_mask=None, q_seq_lens=None,
                    key_cache=None, value_cache=None, token_cache_loc=None, kv_mask=None):
        # B, S, H
        query = mint.reshape(query, (-1, 1, self.n_heads * self.head_dim))

        bs = batch_valid_length.shape[0]

        key = key_cache[token_cache_loc]
        value = value_cache[token_cache_loc]

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
