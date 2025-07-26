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
"""Qwen models' APIs."""
import mindspore.common.dtype as mstype
from mindspore import Tensor, ops, mint, mutable
from mindspore.communication._comm_helper import _is_initialized

import mindspore as ms
import numpy as np
from mindformers.experimental.infer.core.layers import ColumnParallelLinear
from mindformers.models.llama.llama import LlamaPreTrainedModel
from mindformers.modules import Linear
from mindformers.tools.utils import get_predict_run_mode
from mindformers.experimental.infer.models.llama.utils import convert_model_config
from mindformers.parallel_core.inference.parallel_state import (
    get_group_info,
    initialize_model_parallel,
)
from mindformers.parallel_core.inference.utils import (
    get_tp_world_size,
    get_dp_world_size,
)
from mindformers.models.utils import jit


from .transformer import ParallelTransformer

__all__ = ["ParallelQwenForCausalLM"]


class ParallelQwenForCausalLM(LlamaPreTrainedModel):
    r"""
    Provide qwen training loss or logits through network.

    Args:
        config (LlamaConfig): The config of qwen model.

    Returns:
        output: Tensor, the output of llama decoderlayer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=True)
        self.config = convert_model_config(config)

        tp_group = get_group_info('tp').group is None
        dp_group = get_group_info('dp').group is None
        print("tp_group is:{}".format(tp_group))
        print("dp_group is:{}".format(dp_group))
        all_groups_initialized = tp_group and dp_group
        if all_groups_initialized and _is_initialized():
            initialize_model_parallel(tensor_model_parallel_size=self.config.parallel_config.model_parallel,
                                      order='tp-dp')
        print("data_parallel_group:{}".format(get_dp_world_size()))
        print("tensor_model_parallel_group:{}".format(get_tp_world_size()))
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True

        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.slice = ops.StridedSlice()
        self.not_equal = ops.NotEqual()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.ones = ops.Ones()
        self.gather = ops.Gather()
        self.model = ParallelTransformer(config=config)
        if config.parallel_config.vocab_emb_dp:
            self.lm_head = Linear(
                in_channels=config.hidden_size,
                out_channels=config.vocab_size,
                weight_init="normal",
                has_bias=False,
                param_init_type=config.param_init_type,
                compute_dtype=config.compute_dtype
            )
        else:
            self.lm_head = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                config=config.parallel_config,
                bias=False,
                gather_output=True,
                param_init_type=config.param_init_dtype,
                compute_dtype=config.compute_dtype,
            )

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()

        self.npu_mem_size = config.npu_mem_size if hasattr(config, "npu_mem_size") else 2
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.embedding_weight
        self.return_hidden_states = config.return_hidden_states

    def set_dynamic_inputs(self, **kwargs):
        """Prepare inputs for dynamic shape."""
        dynamic_input_ids = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_position_ids = Tensor(shape=[None], dtype=mstype.int64)
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_attention_mask = None
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)

        def get_input():
            if self.npu_mem_size > 0:
                return None
            cache_list = []
            for _ in self.model.layers:
                cache_list.append(Tensor(shape=[None, None, None], dtype=self.config.compute_dtype))
            return mutable(cache_list)

        key_cache = get_input()
        value_cache = get_input()

        dynamic_out_cache_loc = Tensor(shape=[None], dtype=mstype.int64)
        dynamic_token_cache_loc = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_kv_mask = Tensor(shape=[None, 1, 1, None], dtype=mstype.bool_)


        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None,
                            dynamic_prefix_keys_values, None, key_cache, value_cache,
                            dynamic_out_cache_loc, dynamic_token_cache_loc, dynamic_kv_mask)
        else:
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_attention_mask, None, None,
                            dynamic_batch_valid_length, None, None,
                            None, None, dynamic_q_seq_lens, key_cache, value_cache,
                            dynamic_out_cache_loc, dynamic_token_cache_loc, dynamic_kv_mask)

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    @jit
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  prefix_keys_values=None, llm_boost_inputs=None,
                  q_seq_lens=None, key_cache=None, value_cache=None, out_cache_loc=None,
                  token_cache_loc=None, kv_mask=None):
        """
        Forward of qwen model.
        """
        output = self.model(input_ids, batch_valid_length, batch_index, zactivate_len,
                            prefix_keys_values, position_ids=position_ids, attention_mask=attention_mask,
                            q_seq_lens=q_seq_lens, key_cache=key_cache, value_cache=value_cache,
                            out_cache_loc=out_cache_loc, token_cache_loc=token_cache_loc, kv_mask=kv_mask)
        if self.return_hidden_states:
            return output
        pre_gather = self.is_first_iteration and batch_valid_length is not None
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            output = self.gather(output, batch_valid_length - 1, 0)
        logits = self.lm_head(output)

        logits = self.cast(logits, mstype.float32)
        if self.predict_run_mode:
            return self.reshape(logits, (-1, logits.shape[-1]))
        input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
        return logits, input_ids, input_mask
