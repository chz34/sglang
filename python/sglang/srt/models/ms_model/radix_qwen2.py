import os
import time
import logging
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

import torch
from torch import nn

import mindspore as ms
from mindspore import Tensor, JitConfig, Model, mutable
from mindspore.communication.management import get_group_size
from mindspore.communication.comm_func import barrier
from mindspore.nn.utils import no_init_parameters
from mindspore import ops, mint

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead


from mindformers.models import build_network, build_processor, build_tokenizer
from mindformers import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context, build_mf_context
from mindformers.core.context.parallel import ParallelOperator
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.utils import set_strategy_save_path
from mindformers.core.context.validators import execute_validator
from mindformers.models.llama import LlamaConfig as LlamaConfig_MF

from mindformers.parallel_core.inference.utils import get_tp_world_size

from .qwen2_5 import ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF

from .qwen2_weight_processor import Qwen2WeightProcessor
from .utils import tensor_torch2ms, tensor_ms2torch
import inspect
import os

logger = logging.getLogger(__name__)

def load_mf_config_from_file():
    config_path = os.getenv('MF_MODEL_CONFIG', '')
    if not config_path:
        raise ValueError("Please set the environment variable 'MF_MODEL_CONFIG' to the path of the config file.")
    config = MindFormerConfig(os.path.realpath(config_path))
    return config


class RadixModel(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        load_config: LoadConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.model_config = model_config
        self.load_config = load_config

        self.mf_config = load_mf_config_from_file()
        self.mf_config.load_ckpt_format = "safetensors"
        self.mf_config.load_checkpoint = self.model_config.model_path
        self.mf_model_config = self.mf_config.model.model_config
        self.hf_config = self.model_config.hf_config
        self._build_config()
        build_parallel_config(self.mf_config)
        build_context(self.mf_config)

        # if self.mf_config.use_parallel:
        #     self.setup_parallel_context()

        self.build_network()

        self.time_start = time.time()
        self.time_end = time.time()

        self.key_cache = []
        self.value_cache = []

    def _build_config(self):
        """ modify the config accroding to different model """
        self.mf_model_config = LlamaConfig_MF(**self.mf_config.model.model_config)
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
        self.mf_model_config.return_hidden_states = False

        # qwen qkv concat will support in next version
        self.mf_model_config.qkv_concat = False
        setattr(self.mf_model_config, 'npu_mem_size', -1)
        self.mf_config.model.model_config.qkv_concat = False

    def build_network(self):
        default_args = {"parallel_config": self.mf_config.parallel_config,
                        "moe_config": self.mf_config.moe_config}

        self.mf_model_config.parallel_config = self.mf_config.parallel_config
        # Initial network
        with no_init_parameters():  # Delay initialization
            self.network = ParallelQwenForCausalLM_MF(self.mf_model_config)

        self.lm_head = self.network.lm_head

        num_block = self.mf_model_config.num_blocks
        block_size = self.mf_model_config.block_size
        assert(self.mf_model_config.n_kv_heads % get_tp_world_size() == 0)
        num_kv_heads = self.mf_model_config.n_kv_heads // get_tp_world_size()
        head_size = self.mf_model_config.hidden_size // self.mf_model_config.num_heads

    def get_kvcache(self, forward_batch: ForwardBatch):
        if self.key_cache and self.value_cache:
            return mutable(self.key_cache), mutable(self.value_cache)

        for i in range(self.model_config.hf_config.num_hidden_layers):
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(i)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(i)
            #self.key_cache.append(mint.zeros(list(k_cache.shape), dtype=ms.bfloat16))
            #self.value_cache.append(mint.zeros(list(v_cache.shape), dtype=ms.bfloat16))
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

        query_lens_np = forward_batch.seq_lens.cpu().numpy()
        batch_valid_length = forward_batch.seq_lens.cpu().numpy()

        is_finished = [False for _ in range(forward_batch.batch_size)]  # always not finished in the model forward

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
        model_inputs["attention_mask"] = None
        model_inputs["out_cache_loc"] = tensor_torch2ms(forward_batch.out_cache_loc)
        model_inputs["token_cache_loc"] = token_cache_loc
        model_inputs["kv_mask"] = kv_mask
        # print("++++++++model_inputs: ", model_inputs)
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache

        return model_inputs, is_prefill

    def setup_parallel_context(self):
        execute_validator(self.mf_config)
        parallel_operator = ParallelOperator(self.mf_config)
        device_num = get_group_size()
        parallel_operator.parallel_ctx['device_num'] = device_num
        ms.context.reset_auto_parallel_context()
        set_strategy_save_path(parallel_operator.parallel_ctx)
        parallel_operator._set_ms_auto_parallel_context(**parallel_operator.parallel_ctx)
        parallel_operator._set_ms_parallel()


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weight_processor = Qwen2WeightProcessor(self.mf_config, self.network, False)
        weight_processor.load_safetensors_shard(self.mf_config.load_checkpoint)

        self.network.set_dynamic_inputs()
        dynamic_hidden_states = Tensor(shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        self.lm_head.set_inputs(dynamic_hidden_states)

        if self.mf_config.use_parallel:
            barrier()
        logger.info("Model weights loaded successfully.")
        return

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                forward_batch: ForwardBatch):

        # Set the phases and flags
        model_inputs, is_prefill = self.prepare_inputs(input_ids, positions, forward_batch)

        self.time_start = time.time()
        logger.info(f"ms run interval: {self.time_start - self.time_end}")
        if is_prefill:
            self.network.phase = "prefill"
            self.network.add_flags_custom(is_first_iteration=True)
            hidden_states = self.network(**model_inputs)
        else:
            self.network.phase = "increment"
            self.network.add_flags_custom(is_first_iteration=False)
            hidden_states = self.network(**model_inputs)

        if self.mf_model_config.return_hidden_states:
            batch_valid_length = model_inputs["batch_valid_length"]
            pre_gather = is_prefill and batch_valid_length is not None
            if pre_gather:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
                output = ops.Gather()(hidden_states, batch_valid_length - 1, 0)
            else:
                output = hidden_states

            logits = self.lm_head(output)
            logits = logits.view(-1, logits.shape[-1])
        else:
            logits = hidden_states

        self.time_end = time.time()
        print("ms run time: ", self.time_end - self.time_start)

        logits_result = LogitsProcessorOutput(next_token_logits=torch.Tensor(logits.asnumpy()).to(input_ids.device))
        # TODO: npu tensor ms2torch error to be fix
        # logits_result = LogitsProcessorOutput(next_token_logits=tensor_ms2torch(logits))
        return logits_result
