# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import importlib
from enum import Enum
from typing import Callable, Optional
import torch.nn.functional as F
import torch
from vllm.distributed import get_tensor_model_parallel_rank
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
import vllm.envs as envs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_qdq import *

logger = init_logger(__name__)



class CompressedTensorsW4A4MXFP4MoeMethod(CompressedTensorsMoEMethod):

    def __init__(self):
        self.use_marlin = False
        self.group_size = 32

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        self.mask_weights_buffer = None
        self.bt_threshold = 4096

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_weight_bf16 = to_dtype(
            data_lp=layer.w13_weight_packed,
            scale_e8m0=layer.w13_weight_scale,
            elem_dtype=DTYPE_FP4_E2M1,
            block_size=32,
            target_dtype=torch.bfloat16,
            use_fp4_custom_triton_dequant_kernel=False,
            pack_fp6=False,
        )

        w2_weight_bf16 = to_dtype(
            data_lp=layer.w2_weight_packed,
            scale_e8m0=layer.w2_weight_scale,
            elem_dtype=DTYPE_FP4_E2M1,
            block_size=32,
            target_dtype=torch.bfloat16,
            use_fp4_custom_triton_dequant_kernel=False,
            pack_fp6=False,
        )

        delattr(layer, "w13_weight_packed")
        delattr(layer, "w13_weight_scale")
        delattr(layer, "w2_weight_packed")
        delattr(layer, "w2_weight_scale")
 
        layer.register_parameter(
            "w13_weight_unpacked",
            torch.nn.Parameter(
                w13_weight_bf16,
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_weight_unpacked",
            torch.nn.Parameter(
                w2_weight_bf16,
                requires_grad=False,
            ),
        )


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ):
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        num_experts, intermediate_size_per_partition_x2, _ = (
            layer.w13_weight_unpacked.shape
        )
        intermediate_size_per_partition = (
            intermediate_size_per_partition_x2 // 2
        )
        act_fn = F.silu
        num_all_tokens, hidden_dim = x.shape
        num_experts = layer.local_num_experts
        total_num_experts = router_logits.size(-1)
        experts_mask = torch.zeros(
            (x.size(0), total_num_experts), dtype=x.dtype, device=x.device
        )
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)
        experts_mask.scatter_(-1, topk_ids, topk_weights)
        experts_mask = experts_mask.transpose(0, 1)

        mask_weights = torch.zeros(
            (num_all_tokens, total_num_experts),
            dtype=x.dtype,
            device=x.device,
        )
        mask_weights.scatter_(-1, topk_ids, 1)
        mask_weights = mask_weights.transpose(0, 1)
        # Note: ep_size equal tp_size
        ep_rank = get_tensor_model_parallel_rank()
        ep_shift = ep_rank * num_experts
        for expert_index in range(num_experts):
            mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
            current_state_static = x * mask_weight

            local_unpacked_w13 = layer.w13_weight_unpacked[expert_index]
            local_w13_scale = layer.w13_weight_scale[expert_index]

            local_unpacked_w2 = layer.w2_weight_unpacked[expert_index]
            #local_w2_scale = layer.w2_weight_scale[expert_index]

            local_unpacked_w1 = local_unpacked_w13[:intermediate_size_per_partition, ...]
            #half_scale = local_w13_scale.shape[0] // 2
            #local_w1_scale = local_w13_scale[:half_scale, ...]
            local_unpacked_w3 = local_unpacked_w13[intermediate_size_per_partition:, ...]
            #local_w3_scale = local_w13_scale[half_scale:, ...]

            qdq_state = quant_dequant_mxfp4(current_state_static)
            local_w1_out = torch.matmul(qdq_state, local_unpacked_w1.t())

            local_w3_out = torch.matmul(qdq_state, local_unpacked_w3.t())

            w13_out = act_fn(local_w1_out) * local_w3_out

            qdq_w13_out = quant_dequant_mxfp4(w13_out)

            local_w2_out = torch.matmul(qdq_w13_out, local_unpacked_w2.t())
            padded_weight = experts_mask[expert_index + ep_shift].unsqueeze(
                1
            )
            local_w2_out = local_w2_out * padded_weight
            if expert_index == 0:
                final_hidden_states = local_w2_out
            else:
                final_hidden_states += local_w2_out
        return final_hidden_states
