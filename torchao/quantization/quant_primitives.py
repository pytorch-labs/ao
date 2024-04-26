# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import List, Optional, Tuple
import torch
from torch._dynamo import is_compiling as dynamo_is_compiling
from torch._higher_order_ops.out_dtype import out_dtype
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.library import impl

from torchao.kernel.intmm import int_scaled_matmul
from torchao.kernel.intmm import safe_int_mm
from .utils import TORCH_VERSION_AFTER_2_3


_AFTER_TORCH_2_3_ONLY = [
    "per_token_dynamic_quant",
    "get_group_qparams_symmetric",
]

__all__ = [
    "safe_int_mm",
    "dynamically_quantize_per_tensor",
    "quantize_activation_per_token_absmax",
    "dynamically_quantize_per_channel",
    "dequantize_per_tensor",
    "dequantize_per_channel",
    "quant_int8_dynamic_linear",
    "quant_int8_matmul",
    "quant_int8_dynamic_per_token_linear",
    "quant_int8_per_token_matmul",
    "get_groupwise_affine_qparams",
    "pack_tinygemm_scales_and_zeros",
    "unpack_tinygemm_scales_and_zeros",
    "groupwise_affine_quantize_tensor_from_qparams",
    "groupwise_affine_dequantize_tensor_from_qparams",
    "groupwise_affine_quantize_tensor",
    "groupwise_affine_dequantize_tensor",
    # TODO: need to clean up above functions
] + (_AFTER_TORCH_2_3_ONLY if TORCH_VERSION_AFTER_2_3 else [])


_DTYPE_TO_QVALUE_BOUNDS = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.int16: (-(2**15), 2**15 - 1),
    torch.int32: (-(2**31), 2**31 - 1),
}

if TORCH_VERSION_AFTER_2_3:
    _DTYPE_TO_QVALUE_BOUNDS.update({
        torch.uint1: (0, 2**1-1),
        torch.uint2: (0, 2**2-1),
        torch.uint3: (0, 2**3-1),
        torch.uint4: (0, 2**4-1),
        torch.uint5: (0, 2**5-1),
        torch.uint6: (0, 2**6-1),
        torch.uint7: (0, 2**7-1),
    })

# TODO: decide on if we want to allow custom quant_min/quant_max here
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get quant_min and quant_max args based on dtype and also
    verify that they are within the range of possible quant_min/quant_max
    for dtype
    """
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    if quant_min is None:
        quant_min = quant_min_lower_bound
    if quant_max is None:
        quant_max = quant_max_upper_bound

    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    return quant_min, quant_max

def _get_reduction_params(block_size, input_size):
    """Given block_size and input size find the parameters for reduction:

    Output:
        shape_for_reduction: the shape we use to `view` input to prepare it for reduction
        reduction_dims: the dims we'll do reduction over

    Example::
        Input:
          block_size: (3, 3, 2, 10)
          input_size: (3, 3, 10, 10)

        Output:
          shape_for_reduction: (3, 3, 5, 2, 10)
          reduction_dim: [0, 1, 3, 4]
    """
    assert len(block_size) == len(input_size)
    shape_for_reduction = []
    reduction_dims = []
    cur_dim = 0
    for i in range(len(block_size)):
        if block_size[i] != input_size[i] and block_size[i] > 1:
            assert input_size[i] % block_size[i] == 0, f"Expecting input size at {i} dimension: {input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            shape_for_reduction.append(input_size[i] // block_size[i])
            shape_for_reduction.append(block_size[i])
            # reduce over the block_size[i] dim
            reduction_dims.append(cur_dim + 1)
            cur_dim += 2
        else:
            # block_size[i] == input_size[i] or block_size[i] == 1
            shape_for_reduction.append(input_size[i])
            # we only need to reduce over the dimension if block_size is greater than 1
            # otherwise it's already the same as reduced dimension
            if block_size[i] != 1:
                reduction_dims.append(cur_dim)
            cur_dim += 1
    return shape_for_reduction, reduction_dims


def quantize_affine(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None
):
    """
    Args:
      input (torch.Tensor): original float32 or bfloat16 Tensor
      block_size: (List[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
                           e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (float): quantization parameter for affine quantization
      zero_point (int): quantization parameter for affine quantization
      output_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for output Tensor, if not specified, it will be derived from dtype
      quant_max (Optional[int]): maximum quantized value for output Tensor, if not specified, it will be derived from dtype

    Note:
      How can block_size represent different granularities?
      let's say we have a Tensor of size: (3, 3, 10, 10), here is the table showing how block_size represents different
      granularities:

       granularity type       |     block_size
         per_tensor           |    (3, 3, 10, 10)
         per_axis (axis=0)    |    (1, 3, 10, 10)
         per_axis (axis=1)    |    (3, 1, 10, 10)
     per_group (groupsize=2)  |    (3, 3, 10, 2)
     per_group (groupsize=2) for axis = 3 | (3, 3, 2, 10)


    Output:
      quantized tensor with requested dtype
    """
    # TODO: validations
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    quant = torch.clamp(
        torch.round(input / scale) + zero_point, quant_min, quant_max
    ).to(output_dtype)
    quant = quant.view(original_shape)

    return quant

def dequantize_affine(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    *,
    output_dtype: Optional[torch.dtype] = None,
):
    """
    Args:
      input (torch.Tensor): quantized tensor, should match the dtype `dtype` argument
      block_size: (List[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
                               e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (Tensor): quantization parameter for affine quantization
      zero_point (Tensor): quantization parameter for affine quantization
      dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype?): optional dtype for output Tensor, default is fp32

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
    # TODO: validations
    assert input.dtype == input_dtype
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)

    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    dequant = input.to(torch.float32)
    scale = scale.to(torch.float32)
    if zero_point is not None:
        zero_point = zero_point.to(torch.float32)
        dequant -= zero_point
    dequant *= scale
    dequant = dequant.view(original_shape)
    return dequant.to(output_dtype)


class MappingType(Enum):
    SYMMETRIC = 0
    ASYMMETRIC = 1

def choose_qparams_affine(
   input: torch.Tensor,
   mapping_type: MappingType,
   block_size: List[int],
   target_dtype: torch.dtype,
   quant_min: Optional[int] = None,
   quant_max: Optional[int] = None,
   eps: Optional[float] = None,
   scale_dtype: Optional[torch.dtype] = None,
   zero_point_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        input (torch.Tensor): fp32, bf16, fp16 input Tensor
        mapping_type (MappingType): determines how the qparams are calculated, symmetric or asymmetric
        target_dtype (torch.dtype): dtype for target quantized Tensor
        quant_min (Optional[int]): minimum quantized value for target quantized Tensor
        quant_max (Optioanl[int]): maximum quantized value for target quantized Tensor
        eps (Optional[float]: minimum scale
        scale_dtype (torch.dtype): dtype for scales
        zero_point_dtype (torch.dtype): dtype for zero_points

    Output:
        Tuple of scales and zero_points Tensor with requested dtype
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    if scale_dtype is None:
        scale_dtype = torch.float32
    if zero_point_dtype is None:
        zero_point_dtype = torch.float32

    assert len(block_size) == input.dim()
    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    input = input.view(shape_for_reduction)

    if mapping_type == MappingType.SYMMETRIC:
        amax = torch.amax(torch.abs(input), dim=reduction_dims, keepdim=False)
        scale = amax / (float(quant_max - quant_min) / 2)
        zero_point = torch.ones_like(scale)
        zero_point *= int((quant_min + quant_max + 1) / 2)
    elif mapping_type == MappingType.ASYMMETRIC:
        min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input, dim=reduction_dims, keepdim=False)

        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    else:
        raise RuntimeError(f"Unsupported mapping type: {mapping_type}")

    if eps is not None:
        scale = torch.clamp(scale, min=eps)

    return scale.to(dtype=scale_dtype), zero_point.to(dtype=zero_point_dtype)


# copy-pasta of https://www.internalfb.com/intern/anp/view/?id=3350736

def dynamically_quantize_per_tensor(
    x,
    quant_min,
    quant_max,
    target_dtype,
    qscheme=torch.per_tensor_affine,  # for now, reuse existing qscheme enum
):
    # assumes affine quantization

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    if qscheme == torch.per_tensor_affine:
        # get min and max
        # TODO(future): make torch.aminmax work on cpu-half
        # min_val, max_val = torch.aminmax(x)
        min_val = torch.min(x)
        max_val = torch.max(x)

        # calculate scale and zero point based on min and max
        # reference: https://fburl.com/code/srbiybme
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # TODO(future): make torch.clamp with scalar work on cpu-half
        scale = torch.clamp(scale, min=eps).reshape(1)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # quantize based on qmin/qmax/scale/zp
        # reference: torch/ao/quantization/fx/_decomposed.py?lines=63
        quant = torch.clamp(
            torch.round(x / scale) + zero_point, quant_min, quant_max
        ).to(target_dtype)

    else:
        assert qscheme == torch.per_tensor_symmetric, f"unsupported qscheme {qscheme}"
        # assert quant_min == -1 * quant_max, "unsupported quant_min/quant_max"
        amax = torch.max(torch.abs(x))
        scale = amax / (float(quant_max - quant_min) / 2)
        scale = torch.clamp(scale, min=eps).reshape(1)
        quant = torch.clamp(torch.round(x / scale), quant_min, quant_max).to(
            target_dtype
        )
        # do not create a tensor for zero_point as this is expensive
        zero_point = None

    return quant, scale, zero_point


# taken from
# https://github.com/mit-han-lab/smoothquant/blob/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c/smoothquant/fake_quant.py#L26
# and slightly modified


def quantize_activation_per_token_absmax(t):
    n_bits = 8
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]

    scales = t.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max)
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t, scales


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scale is the same dtype as the original tensor
    scale = torch.clamp(scale, min=eps).to(x.dtype)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scale/zp
    # reference: torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(0, 1) / scale
    x_round = torch.round(x_div)
    x_zp = x_round + zero_point
    x_zp = x_zp.transpose(0, 1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scale, zero_point


# reference: https://fburl.com/code/vfsygwd0


def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    y = int_repr.to(out_dtype)
    if zero_point is not None:
        y -= zero_point
    return y * scale


# reference: https://fburl.com/code/org0fmi3


def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=torch.float32):
    # assumes axis is 0
    y = int_repr.transpose(0, 1)
    y = y.to(out_dtype)
    y = y - zero_points
    y = y * scales
    y = y.transpose(0, 1)
    return y


def quant_int8_dynamic_linear(
    x,
    x_quant_min,
    x_quant_max,
    x_q_dtype,
    w_vals_int8_t,
    w_scales,
    w_vals_int8_t_sums_int64,
    bias,
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(
        x, x_quant_min, x_quant_max, x_q_dtype
    )
    # w_vals_int8_t_sums_int64 = w_vals_int8_t.sum(dim=0)
    mm_out = quant_int8_matmul(
        x_vals_int8,
        x_scale,
        x_zp,
        w_vals_int8_t,
        w_vals_int8_t_sums_int64,
        w_scales,
        out_dtype,
    )
    if bias is not None:
        mm_out += bias
    return mm_out


def quant_int8_matmul(
    x_vals_int8,
    x_scale,
    x_zp,
    w_vals_int8_t,
    w_vals_int8_t_sums_int64,
    w_scales,
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # correctness, and things like aligning accumulation behaviors and
    # performance optimizations are left for a future PR.
    # Assumes that weight quantization is symmetric, i.e. w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw (dot(X_i, W_j) - zx * sum(W_j))
    #

    assert x_vals_int8.dtype in (
        torch.uint8,
        torch.int8,
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8_t.dtype == torch.int8
    ), f"w dtype {w_vals_int8_t.dtype} not yet supported"
    assert w_scales.dtype == out_dtype, f"{w_scales.dtype} does not match {out_dtype}"

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    # TODO(before land): add test case for input with bsz
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)
    y_dot_int32 = y_dot_int32.reshape(*x_vals_int8.shape[:-1], -1)

    # TODO(future): consider using integer arithmetic throughout, although
    # TBD if that is actually faster on GPUs
    # need to use 32 bits here to prevent overflow for large shapes,
    # 16 bits is not enough
    y_dot_float32 = y_dot_int32.to(torch.float32)

    #
    # 2. connect it all together
    #

    # mm_unscaled has to stay in float32 for the next two lines to prevent overflow
    mm_unscaled_float32 = y_dot_float32 - (x_zp * w_vals_int8_t_sums_int64)
    y = x_scale * w_scales * mm_unscaled_float32
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y


def quant_int8_dynamic_per_token_linear(
    x,
    w_vals_int8_t,
    w_scales,
    bias,
    out_dtype,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = quant_int8_per_token_matmul(
        x_vals_int8, x_scales, w_vals_int8_t, w_scales, out_dtype
    )
    if bias is not None:
        mm_out += bias
    return mm_out


def quant_int8_per_token_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8_t,
    w_scales,
    output_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # output_dtype. For now, this is written for approximate numerical
    # Assumes that activation and weight quantization are symmetric,
    # i.e. act_zp and w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming output_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw dot(X_i, W_j)
    #

    assert (
        x_vals_int8.dtype == torch.int8
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8_t.dtype == torch.int8
    ), f"w dtype {w_vals_int8_t.dtype} not yet supported"

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #
    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_scaled = int_scaled_matmul(tmp, w_vals_int8_t, x_scales.reshape(-1, 1))

    y = (y_dot_scaled * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
    )

    # can downcast only at the very end
    y = y.to(output_dtype)
    return y


def get_groupwise_affine_qparams(w, n_bit=4, groupsize=128):
    """ """
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)


def pack_tinygemm_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_tinygemm_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def groupwise_affine_quantize_tensor_from_qparams(
    w,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int4x8 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int4x8


def groupwise_affine_dequantize_tensor_from_qparams(
    w_int4x8,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int4x8.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int4x8.shape[-1]
    assert w_int4x8.shape[-1] % groupsize == 0
    assert w_int4x8.dim() == 2

    w_int4x8_grouped = w_int4x8.reshape(-1, groupsize)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    w_dq = (
        w_int4x8_grouped.sub(2 ** (n_bit - 1))
        .mul(scales)
        .add(zeros)
        .reshape_as(w_int4x8)
    )
    return w_dq


def groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_groupwise_affine_qparams(w, n_bit, groupsize)
    w_int4x8 = groupwise_affine_quantize_tensor_from_qparams(
        w, scales, zeros, n_bit, groupsize
    )
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
    return w_int4x8, scales_and_zeros


def groupwise_affine_dequantize_tensor(
    w_int4x8,
    scales_and_zeros,
    n_bit=4,
    groupsize=128,
):
    scales, zeros = unpack_tinygemm_scales_and_zeros(scales_and_zeros)
    return groupwise_affine_dequantize_tensor_from_qparams(
        w_int4x8, scales, zeros, n_bit, groupsize
    )


# TODO: replace this with torch.ao.quantization.PerChannelMinMaxObserver
def get_group_qparams_symmetric(w, n_bit=4, groupsize=128, precision=torch.float32):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    max_val_abs = torch.max(-min_val_neg, max_val_pos)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))

    scales = max_val_abs / (float(max_int - min_int) / 2)
    scales = torch.max(scales, torch.full_like(scales, torch.finfo(torch.float32).eps))
    # TODO: make sure abs(scales) is not too small?
    zeros = torch.full_like(scales, 0)
    return scales.to(precision).reshape(w.shape[0], -1), zeros.to(precision).reshape(
        w.shape[0], -1
    )


if TORCH_VERSION_AFTER_2_3:
    def group_quantize_tensor_symmetric(
        w,
        n_bit=4,
        group_size=128,
        precision=torch.float32,
    ):
        scales, zeros = get_group_qparams_symmetric(w, n_bit, group_size, precision)
        n_bit = 4
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        # TODO: currently we don't know how to express torch.int4, we'll
        # add torch.int4 to core later
        w_int8 = torch.ops.quantized_decomposed.quantize_per_channel_group(
            w, scales, zeros, min_int, max_int, torch.int8, group_size
        )

        return w_int8, scales, zeros


    def down_size(size):
        assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
        return (*size[:-1], size[-1] // 2)


    def up_size(size):
        return (*size[:-1], size[-1] * 2)


    quantized_decomposed_lib.define("pack_int4_from_int8(Tensor int8_data) -> Tensor")


    @impl(quantized_decomposed_lib, "pack_int4_from_int8", "CompositeExplicitAutograd")
    def pack_int4_from_int8(int8_data: torch.Tensor) -> torch.Tensor:
        # converting to uint8 for operations
        shape = int8_data.shape
        assert shape[-1] % 2 == 0
        int8_data = int8_data.contiguous().view(-1)
        return (int8_data[::2] << 4 | int8_data[1::2]).view(down_size(shape))


    quantized_decomposed_lib.define("unpack_int4_to_int8(Tensor int8_data) -> Tensor")


    @impl(quantized_decomposed_lib, "unpack_int4_to_int8", "CompositeExplicitAutograd")
    def unpack_int4_to_int8(int8_data: torch.Tensor) -> torch.Tensor:
        """Get the original weight from the normalized float weight format"""
        # since we are using int8 we will decode 2 entries per byte
        # Shift elements down 4 and select out the bottom 4 bits
        shape = int8_data.shape
        first_elements = (int8_data >> 4).to(torch.int8)
        second_elements = (int8_data & 0b1111).to(torch.int8)
        return torch.stack([first_elements, second_elements], dim=-1).view(up_size(shape))


    def per_token_dynamic_quant(input: torch.Tensor) -> torch.Tensor:
        orig_dtype = input.dtype
        # TODO: we may need to make the choose_qparams op configurable
        (
            scales,
            zero_points,
        ) = torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric(
            input, torch.int8
        )

        # TODO: get these from torch.int8
        quant_min = -128
        quant_max = 127
        input = torch.ops.quantized_decomposed.quantize_per_token(
            input, scales, zero_points, quant_min, quant_max, torch.int8
        )
        input = torch.ops.quantized_decomposed.dequantize_per_token(
            input, scales, zero_points, quant_min, quant_max, torch.int8, orig_dtype
        )
        return input.to(orig_dtype)
