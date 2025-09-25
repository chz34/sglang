# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from typing import Any, Dict, Optional

import mindspore as ms
import torch

from sglang.srt.distributed import get_tp_group, get_world_group


def tensor_torch2ms(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor):
        return x

    if x.device.type == "cpu":
        # TODO: dlpack support CPU, for now will slow down the weight loading
        if x.dtype == torch.bfloat16:
            return ms.Tensor(
                x.contiguous().to(torch.float32).numpy(), dtype=ms.bfloat16
            )
        return ms.Tensor(x.contiguous().numpy())

    # torch tensor -> dlpack -> mindspore tensor
    pt_dlpack = torch.utils.dlpack.to_dlpack(x)
    ms_tensor = ms.Tensor.from_dlpack(pt_dlpack)
    return ms_tensor


def tensor_ms2torch(x: ms.Tensor):
    if x is None or not isinstance(x, ms.Tensor):
        return x

    if x.device == "CPU":  # TODO: dlpack support CPU
        if x.dtype == ms.bfloat16:
            return torch.tensor(
                x.contiguous().to(ms.float32).asnumpy(), dtype=torch.bfloat16
            )
        return torch.tensor(x.contiguous().asnumpy())

    # ms tensor -> dlpack -> torch tensor
    ms_dlpack = x.to_dlpack()
    torch_tensor = torch.from_dlpack(ms_dlpack)
    return torch_tensor


def split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size):
    if shard_dim is None:
        loaded_weight = loaded_weight[:]
        return loaded_weight

    end_idx = start_idx + shard_size
    if shard_dim == 0:
        loaded_weight = loaded_weight[start_idx:end_idx]
    elif shard_dim == 1:
        loaded_weight = loaded_weight[:, start_idx:end_idx]
    elif shard_dim == 2:
        loaded_weight = loaded_weight[:, :, start_idx:end_idx]
    else:
        raise ValueError("shard_dim:{} is not supported.".format(shard_dim))
    return loaded_weight


def _get_tp_group_name():
    return get_tp_group().unique_name


def _get_world_group_name():
    return get_world_group().unique_name


def set_weight_attrs(
    weight: ms.Parameter,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)
