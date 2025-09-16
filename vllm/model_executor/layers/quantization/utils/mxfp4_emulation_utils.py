import torch
import vllm.envs as envs

from torchao.prototype.mx_formats.mx_tensor import (
    to_mx,
    to_dtype,
    ScaleCalculationMode,
)




def fp4_121_positive(x:torch.Tensor, stochastic_rounding:bool=False) -> torch.Tensor:
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)

    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def ue5m3(x:torch.Tensor) -> torch.Tensor:
    # NOTE: Assume that array values are in [0, 114688]. (14*2**13 = 114688)
    mask = x <= 2**(-17)
    x_1 = x * mask
    x_2 = x * (~mask) + torch.ones_like(x) * mask

    x_1 = torch.round(x_1 / 2**(-17)) * (2**(-17))

    e = torch.floor(torch.log2(x_2)) - 3
    s = 2**e
    x_2 = torch.round(x_2 / s) * s

    return x_1 * mask + x_2 * (~mask)


FP8_E4M3_MAX = 240.0
FP8_E4M3_MAX = 448.0
def fp4_121_scaled_old(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   scale_format:str='e8m0') -> torch.Tensor:
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    if scale_format == 'e8m0':
        scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))

    elif scale_format == 'e4m3':
        from habana_frameworks.torch import core as htcore
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t

        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b, 1.0, False, False, torch.float8_e4m3fn)[0]
        htcore.mark_step()
        up_cast = down_cast.to(scale_per_b.dtype)
        scale_per_b = up_cast
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)

        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t

    elif scale_format == 'ue5m3':
        UE5M3_MAX = 114688.0
        nvfp4_max = fp4_121_max * UE5M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t

        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]

        scale_per_b = ue5m3(fp4_121_max / scale_per_b)

        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)

        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t


    else: # scale_format == 'bf16'
        scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)

    x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
    return sign * x_fp4_abs

def fp4_121_scaled_even_rounding(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   scale_format:str='e8m0') -> torch.Tensor:
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    assert scale_format == 'e8m0', f"Unsupported scale format: {scale_format}"
    if scale_format == 'e8m0':
        # scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
        amax_x = x_abs.max(dim=-1, keepdim=True)[0]
        scale_tmp = torch.floor(torch.log2(amax_x)) - 2.0
        scale_clamp = torch.clamp(scale_tmp, min=-127, max=127)
        scale = torch.pow(2.0, scale_clamp)

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)

    x_fp4_abs = fp4_121_positive(x_abs / scale, stochastic_rounding) * scale
    return sign * x_fp4_abs

fp4_121_scaled = fp4_121_scaled_old



def run_mxfp4_emulations(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
    group_size = 32
    # quantize input to (FP4 and interleaved block scale)
    input_scale, x_q = to_mx(
        data_hp=x,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
        pack_fp6=False,
    )

    # dequantize input
    x_dq = to_dtype(
        data_lp=x_q,
        scale_e8m0=input_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
    )

    # dequantize weight
    w_dq = to_dtype(
        data_lp=weight,
        scale_e8m0=weight_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out



def run_mxfp4_emulations_unpacked_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
):
    group_size = 32
    # quantize input to (FP4 and interleaved block scale)
    input_scale, x_q = to_mx(
        data_hp=x,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
        pack_fp6=False,
    )

    # dequantize input
    x_dq = to_dtype(
        data_lp=x_q,
        scale_e8m0=input_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
    )

    # dequantize weight
    w_dq = weight

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out
