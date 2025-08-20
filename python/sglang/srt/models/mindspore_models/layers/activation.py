# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

from mindspore import Tensor, nn, ops

class SwiGLU(nn.Cell):
    """ An activation function for SwiGLU

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return (num_tokens, d) or (batch_size, seq_len, d)
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.silu = nn.SiLU()
        self.split = ops.auto_generate.SplitWithSize()
        self.mul = ops.Mul()
        
    def construct(self, x: Tensor) -> Tensor:
        hidden_size = x.shape[-1] // 2
        size = [hidden_size, hidden_size]
        gate, up = self.split(x, size, dim=-1)
        gate = self.silu(gate)
        hidden = self.mul(up, gate)
        return hidden
