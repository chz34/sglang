from typing import List, Optional

import mindspore as ms

from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.models.mindspore_models.utils import set_weight_attrs


class W8A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Int8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        layer.weight = ms.Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = ms.Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ms.Parameter(
            ms.mint.zeros(
                (sum(output_partition_sizes), input_size_per_partition), dtype=ms.int8
            ),
            requires_grad=False,
            weight_loader=weight_loader,
        )
        layer.weight = weight

        weight_scale = ms.Parameter(
            ms.mint.zeros((sum(output_partition_sizes), 1), dtype=ms.float32),
            requires_grad=False,
            weight_loader=weight_loader,
        )
        layer.weight_scale = weight_scale

        self.matmul = ms.ops.QuantBatchMatmul(
            transpose_x1=False, transpose_x2=True, dtype=params_dtype
        )
        self.quant = ms.ops.QuantV2()

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ):
        weight = layer.weight
        deq_scale = layer.deq_scale
        input_scale = layer.input_scale
        input_offset = layer.input_offset
        qx = self.quant(x, input_scale, input_offset, False, "ROUND", ms.dtype.int8)
        output_shape = qx.shape[:-1] + (self.output_size_per_partition,)
        qx = qx.reshape(-1, self.input_size_per_partition)
        qx = self.matmul(qx, weight, deq_scale, None, layer.quant_bias, None)
        if bias is not None:
            qx = ms.mint.add(qx, bias)
        qx = qx.reshape(output_shape)

        return qx


class MSW8A8LinearMethod(LinearMethodBase):
    """Linear method for NPU quantization.

    This class search for specific quantization
    implementation supported on NPU hardware for linear methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: ms.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        """
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=params_dtype)

        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        """

        q_weight_dict = {
            "weight": ms.mint.zeros(
                (sum(output_partition_sizes), input_size_per_partition), dtype=ms.int8
            ),
            "input_scale": ms.mint.zeros(1, dtype=ms.float32),
            "input_offset": ms.mint.zeros(1, dtype=ms.float32),
            "quant_bias": ms.mint.zeros(output_size_per_partition, dtype=ms.int32),
            "deq_scale": ms.mint.zeros(
                output_size_per_partition,
                dtype=ms.float32 if params_dtype == ms.bfloat16 else ms.int64,
            ),
            "weight_scale": ms.mint.zeros(
                output_size_per_partition, 1, dtype=params_dtype
            ),
            "weight_offset": ms.mint.zeros(
                output_size_per_partition, 1, dtype=params_dtype
            ),
        }

        for name, data in q_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        self.quant = ms.ops.QuantV2()
        self.matmul = ms.ops.QuantBatchMatmul(
            transpose_x1=False, transpose_x2=True, dtype=params_dtype
        )

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        return
        # expanding_factor = layer.weight.data.shape[1]
        # layer.aclnn_input_scale = ms.Parameter(
        #     layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
        #     requires_grad=False,
        # )
        # layer.aclnn_input_scale_reciprocal = 1 / ms.Parameter(
        #     layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
        #     requires_grad=False,
        # )
        # layer.aclnn_input_offset = ms.nn.Parameter(
        #     layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
        #     requires_grad=False,
        # )
        # if self.transpose_weight:
        #     layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # layer.weight_scale.data = ms.flatten(layer.weight_scale.data)
        # layer.weight_offset.data = ms.flatten(layer.weight_offset.data)
        # layer.weight.data = ms.ops.format_cast(layer.weight.data, 29)

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        from sglang.srt.models.mindspore_models.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != ms.int8:
            qx = self.quant(
                x, layer.input_scale, layer.input_offset, False, "ROUND", ms.dtype.int8
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = ms.mint.zeros_like(layer.quant_bias)
        else:
            quant_bias = layer.quant_bias
        return self.matmul(
            qx,
            layer.weight,
            layer.deq_scale,
            None,
            quant_bias,
            None,
        )
