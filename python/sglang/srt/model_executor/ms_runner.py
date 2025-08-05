#!/usr/bin/env python3
# encoding: utf-8
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
"""ms_runner launch MindSpore distributed modules."""

import os
import torch
import subprocess
import sys
from pathlib import Path

from sglang.srt.distributed.parallel_state import _groups
import mindspore as ms
from mindspore.communication import create_group
from mindspore._c_expression import GroupOptions

class _Tmp:
    def __init__(self):
        self.sched_p = None

    def set_sched_process(self, p):
        self.sched_p = p

    def __del__(self):
        if self.sched_p:
            self.sched_p.kill()


_tmp = _Tmp()

def _get_host_and_ip(distributed_init_method):
    try:
        _, ip_str, port_str = distributed_init_method.split(":")
        ip = ip_str.split("/")[-1]
        port = int(port_str)
    except Exception as e:
        raise RuntimeError(
            "Cannot get host and port information from %s, error: %s!"
            % (distributed_init_method, str(e))
        )

    return ip, port

def set_ms_parallel_env(rank, local_rank, world_size, init_method):
    if not os.getenv("MS_ROLE"):
        # Not call from msrun, should call a subprocess for scheduler.
        if rank == 0:
            with open(str(Path() / "schedule.log"), "w") as scheduler_f:
                script = Path(__file__).parent / "scheduler_init.py"
                sched_p = subprocess.Popen(
                    [
                        sys.executable,
                        str(script),
                        "--rank_id",
                        str(rank),
                        "--rank_size",
                        str(world_size),
                        "--distributed_init_method",
                        str(init_method),
                    ],
                    shell=False,
                    stdout=scheduler_f,
                    stderr=subprocess.STDOUT,
                )
                global _tmp
                _tmp.set_sched_process(sched_p)

        os.environ["DEVICE_ID"] = str(local_rank)
        os.environ["MS_WORKER_NUM"] = str(world_size)
        os.environ["MS_ROLE"] = "MS_WORKER"
        os.environ["MS_NODE_ID"] = str(rank)
        comm_addr, comm_port = _get_host_and_ip(init_method)
        os.environ["MS_SCHED_HOST"] = str(comm_addr)
        os.environ["MS_SCHED_PORT"] = str(comm_port)

def reuse_hccl_comm():
    for group_name, group in _groups.items():
        # Torch ProcessGroupHccl
        device_group = group().device_group
        hccl_comm_handle = device_group._get_backend(torch.device("npu")).get_hccl_comm(group().local_rank)
        print(f"MindSpore reuse torch group: {device_group}, group_name: {group_name}, local rank: {group().local_rank},"
              f"hccl communicator handle: {hex(hccl_comm_handle)}", flush=True)
        # Create MS communication group by hccl comm handle to reuse Torch group.
        group_options = GroupOptions()
        group_options.hccl_config = {"hccl_comm": hccl_comm_handle}
        create_group(group_name, group().ranks, group_options)

def init_ms_distributed(world_size, rank, local_rank, server_args, port):
        if server_args.dist_init_addr:
            dist_init_method = f"tcp://{server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://{server_args.host}:{port + 33}"
        set_ms_parallel_env(rank, local_rank, world_size, dist_init_method)

        ms.set_context(infer_boost="on", jit_level="O0")
        ms.set_context(mode=ms.context.PYNATIVE_MODE)
        ms.set_device("Ascend", local_rank)
        ms.communication.init("hccl")
        # After distributed job is initialized, reuse hccl comms for MindSpore.
        reuse_hccl_comm()
