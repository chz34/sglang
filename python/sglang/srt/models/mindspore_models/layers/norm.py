# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from typing import Optional, Tuple, Type, Union
from mindspore import Parameter, Tensor, dtype, mint, nn, ops

class RMSNorm(nn.Cell):
    def __init__(self, norm_dim: int, eps: float, param_dtype: Optional[Type]) -> None:
        super().__init__()

        self.weight = Parameter(mint.ones(norm_dim, dtype=param_dtype))
        self.rms_norm = ops.RmsNorm(eps)

    def construct(
        self, x: Tensor, residual: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x
        output = self.rms_norm(x, self.weight)[0]
        if residual is None:
            return output
        return output, residual
