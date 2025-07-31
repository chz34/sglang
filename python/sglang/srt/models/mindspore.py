import logging
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np

import torch
# from torch import nn

import mindspore as ms
from mindspore import nn
from mindspore import Tensor, mutable

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.distributed import get_tp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from sglang.srt.models.ms_model.qwen3 import Qwen3ForCausalLM

from mindspore.communication import create_group
from mindspore._c_expression import GroupOptions

from sglang.srt.models.ms_model.utils import tensor_torch2ms, tensor_ms2torch

type_model_map = {
    "qwen3": Qwen3ForCausalLM,
}

logger = logging.getLogger(__name__)
class MindSporeForCausalLM(torch.nn.Module):
    def __init__(self,
                config: Any,
                quant_config: Optional[QuantizationConfig] = None,
                prefix: str = "",) -> None:
        super().__init__()
        self.config = config

        ms.set_context(infer_boost="on", jit_level="O0")
        ms.set_context(mode=ms.context.PYNATIVE_MODE)
        ms.set_context(graph_kernel_flags="--disable_pass=gather_pre_rms_norm_fusion")

        logger.info(
            "MindSporeForCausalLM tp size %d tp rank %d",
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank()
        )
        # ms.set_device("Ascend", get_attention_tp_rank())
        ms.communication.init("hccl")
        tp = get_tp_group()
        tp_group_name = tp.unique_name
        self.group_name = tp_group_name
        device_group = tp.device_group

        self.local_rank = torch.distributed.get_rank(device_group)
        # self.local_rank = test_group.local_rank
        hccl_comm_handle = device_group._get_backend(torch.device("npu")).get_hccl_comm(self.local_rank)
        group_options = GroupOptions()
        group_options.hccl_config = {"hcom": hccl_comm_handle}
        create_group(tp_group_name, tp.ranks, group_options)
        logger.info("TP Group created")

        model_type = self.config.model_type
        if model_type not in type_model_map:
            raise ValueError(f"Unsupported arch {arch}")
        arch = type_model_map[model_type]
        self.model = arch(config=config, quant_config=quant_config)

        self.lower_triangle_mask = Tensor(
            np.triu(np.ones(shape=(128, 128)), 1), dtype=self.config.param_dtype
        )
        self.key_cache = []
        self.value_cache = []

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)

    def get_kvcache(self, forward_batch: ForwardBatch):
        if self.key_cache and self.value_cache:
            return mutable(self.key_cache), mutable(self.value_cache)

        for i in range(self.config.num_hidden_layers):
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(i)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(i)

            #token_seq, num_head, head_dim = k_cache.shape
            #self.key_cache.append(mint.zeros(list((token_seq, num_head * 1, head_dim)), dtype=ms.bfloat16))
            #self.value_cache.append(mint.zeros(list((token_seq, num_head * 1, head_dim)), dtype=ms.bfloat16))
            self.key_cache.append(tensor_torch2ms(k_cache))
            self.value_cache.append(tensor_torch2ms(v_cache))

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

        return tensor_torch2ms(token_cache_loc), tensor_torch2ms(kv_mask)

    def prepare_inputs(self, input_ids, positions, forward_batch):
        key_cache, value_cache = self.get_kvcache(forward_batch)

        is_prefill = forward_batch.forward_mode.is_extend()
        is_prefill = is_prefill and forward_batch.extend_prefix_lens.sum().item() == 0

        query_lens_np = forward_batch.seq_lens.cpu().numpy()
        batch_valid_length = forward_batch.seq_lens.cpu().numpy()

        if is_prefill:
            q_seq_lens = query_lens_np
        else:
            q_seq_lens = np.ones([forward_batch.batch_size], dtype=np.int32)

        token_cache_loc, kv_mask = self.prepare_token_cache_loc_with_mask(forward_batch)

        model_inputs = {}
        model_inputs["input_ids"] = tensor_torch2ms(input_ids).to(ms.int32)
        model_inputs["batch_valid_length"] = ms.Tensor(batch_valid_length, dtype=ms.int32)
        model_inputs["position_ids"] = tensor_torch2ms(positions)
        model_inputs["q_seq_lens"] = ms.Tensor(q_seq_lens, dtype=ms.int32)
        model_inputs["attention_mask"] = self.lower_triangle_mask
        model_inputs["out_cache_loc"] = tensor_torch2ms(forward_batch.out_cache_loc)
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
        logits = self.model(**model_inputs)
        logits_result = LogitsProcessorOutput(
            next_token_logits=torch.Tensor(logits.asnumpy()).to(input_ids.device)
        )
        # TODO: npu tensor ms2torch error to be fix
        # logits_result = LogitsProcessorOutput(next_token_logits=tensor_ms2torch(logits))
        return logits_result


EntryClass = [MindSporeForCausalLM]