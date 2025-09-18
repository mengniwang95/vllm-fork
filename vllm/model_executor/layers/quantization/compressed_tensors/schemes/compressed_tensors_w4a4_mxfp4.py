# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import (  # noqa: E501
    run_mxfp4_emulations,
    run_mxfp4_emulations_unpacked_weight,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.utils.mxfp4_qdq import *

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4MXFp4"]


class CompressedTensorsW4A4MXFp4(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = 32

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                # dtype=torch.uint8,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        #weight_bf16 = to_dtype(
        #    data_lp=layer.weight_packed,
        #    scale_e8m0=layer.weight_scale,
        #    elem_dtype=DTYPE_FP4_E2M1,
        #    block_size=32,
        #    target_dtype=torch.bfloat16,
        #    use_fp4_custom_triton_dequant_kernel=False,
        #    pack_fp6=False,
        #)
        weight_fp8, scale = dequant_mxfp4_to_fp8(layer.weight_packed, layer.weight_scale)
        weight_bf16 = mxfp4_fp8_weight_to_bf16(weight_fp8, scale)

        del layer.weight_packed
        del layer.weight_scale
        layer.register_parameter(
            "weight_unpacked",
            torch.nn.Parameter(
                weight_bf16,
                requires_grad=False,
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_qdq = quant_dequant_mxfp4(x)
        out = torch.matmul(x_qdq, layer.weight_unpacked.t())
        return out
