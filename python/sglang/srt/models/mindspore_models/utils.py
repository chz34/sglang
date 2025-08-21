# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import mindspore as ms
import torch

from sglang.srt.distributed import get_tp_group

FORMAT_TYPE = {
    "nz": 29,  # TODO: need a variable or enum from mindspore to keep consistency
}


def tensor_torch2ms(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor):
        return x

    # if x.device.type == "cpu":
    # TODO: dlpack support CPU, for now will slow down the weight loading
    if x.dtype == torch.bfloat16:
        return ms.Tensor(x.contiguous().to(torch.float32).numpy(), dtype=ms.bfloat16)
    return ms.Tensor(x.contiguous().cpu().numpy())

    # torch tensor -> dlpack -> mindspore tensor
    pt_dlpack = torch.utils.dlpack.to_dlpack(x)
    ms_tensor = ms.Tensor.from_dlpack(pt_dlpack)
    return ms_tensor


def format_cast(x: ms.Tensor, format: str):
    if format in FORMAT_TYPE:
        return ms.ops.auto_generate.format_cast(x, FORMAT_TYPE[format])
    else:
        raise ValueError(f"Unknown format {format}")


def tensor_ms2torch(x: ms.Tensor):
    if x is None or not isinstance(x, ms.Tensor):
        return x

    # if x.device == "CPU":  # TODO: dlpack support CPU
    if x.dtype == ms.bfloat16:
        return torch.tensor(
            x.contiguous().to(ms.float32).asnumpy(), dtype=torch.bfloat16
        )
    return torch.tensor(x.contiguous().asnumpy())

    # ms tensor -> dlpack -> torch tensor
    ms_dlpack = x.to_dlpack()
    torch_tensor = torch.from_dlpack(ms_dlpack)
    return torch_tensor


def _get_tp_group_name():
    return get_tp_group().unique_name


def get_ascend_soc_version():
    from mindspore._c_expression import MSContext

    return MSContext.get_instance().get_ascend_soc_version()


def is_310p():
    device = get_ascend_soc_version()
    return device in ["310p", "ascend310p"]
