import mindspore as ms
import torch

def tensor_pt2ms(x: torch.Tensor):
    if x.dtype == torch.bfloat16:
        return ms.Tensor(x.cpu().to(torch.float32).numpy(), dtype=ms.bfloat16)
    else:
        return ms.Tensor(x.cpu().numpy())