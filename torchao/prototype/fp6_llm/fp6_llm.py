from functools import reduce
import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.prototype.custom_fp_utils import _f32_to_fpx_unpacked, _fpx_unpacked_to_f32, _n_ones
from torchao.prototype.mx_formats.constants import F6_E3M2_MAX
from torchao.ops import quant_llm_linear
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE


def _pack(x: Tensor, n_bits: int) -> Tensor:
    return reduce(torch.bitwise_or, [x[..., i::(8 // n_bits)] << (8 - (i + 1) * n_bits) for i in range(8 // n_bits)])


def _unpack(x: Tensor, n_bits: int) -> Tensor:
    return torch.stack([(x >> (8 - (i + 1) * n_bits)) & ((1 << n_bits) - 1) for i in range(8 // n_bits)], dim=-1).flatten(-2)


# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h#L87-L116
def _bit_interleave(x: Tensor, n_bits: int, undo: bool = False) -> Tensor:
    # the original code unpacks/packs the values from/to uint32 while we unpack/pack the values from/to uint8
    # thus, we need to reverse byte order within a uint32 word.
    x = x.reshape(-1, 4).flip(1)

    x = _unpack(x, n_bits)
    x = x.view(-1, 4 * (8 // n_bits))

    bit_order = {
        1: [1, 5, 9, 13, 17, 21, 25, 29, 3, 7, 11, 15, 19, 23, 27, 31,
            0, 4, 8, 12, 16, 20, 24, 28, 2, 6, 10, 14, 18, 22, 26, 30],
        2: [1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14],
        4: [1, 5, 3, 7, 0, 4, 2, 6],
    }[n_bits]

    if undo:
        bit_order = [bit_order.index(i) for i in range(len(bit_order))]

    x = x[:, bit_order]
    x = _pack(x, n_bits)

    # reverse byte order within a uint32 word again.
    x = x.reshape(-1, 4).flip(1)
    return x.flatten()


# this is a literal adaptation of FP6-LLM ahead-of-time bit-level pre-packing
# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h
def _to_tc_fpx(tensor: Tensor, ebits: int, mbits: int) -> Tensor:
    assert tensor.ndim == 2
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor_fpx = _f32_to_fpx_unpacked(tensor.float(), ebits, mbits)

    # Pass 1 from original code
    tensor_fpx = tensor_fpx.view(M // 64, 4, 2, 8, N // 16, 2, 8)
    tensor_fpx = tensor_fpx.permute(0, 4, 1, 5, 2, 3, 6)
    tensor_fpx = tensor_fpx.reshape(-1, 32, 2)
    tensor_fpx = tensor_fpx.permute(1, 0, 2)
    tensor_fpx = tensor_fpx.flatten()

    total_bits = 1 + ebits + mbits
    n_used = 0
    fragments = []

    for y in [1, 2, 4]:
        if total_bits & y:
            mask = (1 << y) - 1
            tensor_ybit = (tensor_fpx >> (total_bits - n_used - y)) & mask
            tensor_ybit = _pack(tensor_ybit, y)

            tensor_ybit = tensor_ybit.view(32, -1, 4).permute(1, 0, 2).flip(2)  # Pass 2 from original code
            tensor_ybit = _bit_interleave(tensor_ybit.flatten(), y)             # Pass 3 from original code
            fragments.append(tensor_ybit)
            n_used += y

    return torch.cat(fragments, dim=0).view(M, -1)


def _to_scaled_tc_fpx(tensor: Tensor, ebits: int, mbits: int) -> Tuple[Tensor, Tensor]:
    exp_bias = _n_ones(ebits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2 ** mbits))

    scale = tensor.abs().amax(1).clamp(min=1e-12) / max_normal
    tc_fpx_tensor = _to_tc_fpx(tensor / scale.view(-1, 1), ebits, mbits)
    return tc_fpx_tensor, scale.half()


# more optimized version of _to_tc_float6_e3m2_original() by merging ops
# https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/utils/weight_prepacking.h
def to_tc_float6_e3m2(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor_fp6 = _f32_to_fpx_unpacked(tensor.float(), 3, 2)
    tensor_fp6 = tensor_fp6.view(M // 64, 2, 2, 2, 8, N // 16, 2, 8)
    tensor_fp6 = tensor_fp6.flip(3)

    tensor_2bit = (tensor_fp6 >> 4) & 0b11
    tensor_2bit = tensor_2bit.permute(0, 5, 1, 4, 7, 3, 2, 6)
    tensor_2bit = _pack(tensor_2bit.flatten(), 2)

    tensor_4bit = tensor_fp6 & 0b1111
    tensor_4bit = tensor_4bit.permute(0, 5, 1, 2, 4, 7, 3, 6)
    tensor_4bit = _pack(tensor_4bit.flatten(), 4)

    return torch.cat([tensor_2bit, tensor_4bit], dim=0).view(M, -1)


def to_scaled_tc_float6_e3m2(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    scale = F6_E3M2_MAX / tensor.abs().amax(1).clamp(min=1e-12)
    tc_fp6_tensor = to_tc_float6_e3m2(tensor * scale.view(-1, 1))
    return tc_fp6_tensor, scale.reciprocal().half()


def _from_tc_fpx(tensor: Tensor, ebits: int, mbits: int) -> Tensor:
    total_bits = 1 + ebits + mbits
    M = tensor.shape[0]
    size = tensor.numel()
    tensor = tensor.flatten()
    offset = 0
    n_used = 0

    tensor_fpx = None

    for y in [1, 2, 4]:
        if total_bits & y:
            size_ybit = size // total_bits * y
            tensor_ybit = tensor[offset : offset + size_ybit]
            offset += size_ybit

            tensor_ybit = _bit_interleave(tensor_ybit, y, undo=True)            # undo Pass 3
            tensor_ybit = tensor_ybit.view(-1, 32, 4).flip(2).permute(1, 0, 2)  # undo Pass 2

            tensor_ybit = _unpack(tensor_ybit.flatten(), y)
            tensor_ybit = tensor_ybit << (total_bits - n_used - y)
            n_used += y

            if tensor_fpx is None:
                tensor_fpx = tensor_ybit
            else:
                tensor_fpx |= tensor_ybit

    # undo Pass 1
    tensor_fpx = tensor_fpx.view(32, -1, 2).permute(1, 0, 2)
    tensor_fpx = tensor_fpx.reshape(M // 64, -1, 4, 2, 2, 8, 8)
    tensor_fpx = tensor_fpx.permute(0, 2, 4, 5, 1, 3, 6)
    tensor_fpx = tensor_fpx.reshape(M, -1)

    tensor_fp32 = _fpx_unpacked_to_f32(tensor_fpx, ebits, mbits)
    return tensor_fp32


def from_tc_float6_e3m2(tensor: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    N = tensor.shape[1] // 3 * 4
    assert (M % 64 == 0) and (N % 64 == 0)
    size_2bit = M * N // 4
    size_4bit = M * N // 2
    tensor = tensor.view(-1)
    assert tensor.numel() == size_2bit + size_4bit

    tensor_2bit, tensor_4bit = tensor.split([size_2bit, size_4bit])

    tensor_2bit = _unpack(tensor_2bit, 2)
    tensor_2bit = tensor_2bit.view(M // 64, N // 16, 2, 8, 8, 2, 2, 2)
    tensor_2bit = tensor_2bit.permute(0, 2, 6, 5, 3, 1, 7, 4)

    tensor_4bit = _unpack(tensor_4bit, 4)
    tensor_4bit = tensor_4bit.view(M // 64, N // 16, 2, 2, 8, 8, 2, 2)
    tensor_4bit = tensor_4bit.permute(0, 2, 3, 6, 4, 1, 7, 5)

    tensor_fp6 = (tensor_2bit << 4) | tensor_4bit
    tensor_fp6 = tensor_fp6.flip(3).reshape(M, N)
    return _fpx_unpacked_to_f32(tensor_fp6, 3, 2).to(dtype)


# https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
_SPLIT_K_MAP = [
    {  # tokens: [1, 64]
        3072: 18,
        4096: 13,
        5120: 10,
        6144: 9,
        8192: 6,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 7
    },
    {  # tokens: [65:128]
        3072: 9,
        4096: 6,
        5120: 5,
        6144: 9,
        8192: 3,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 6
    },
    {  # tokens: [129:192]
        3072: 6,
        4096: 4,
        5120: 7,
        6144: 3,
        8192: 2,
        10240: 5,
        14336: 5,
        28672: 5,
        57344: 4
    },
    {  # tokens: [193:256]
        3072: 9,
        4096: 3,
        5120: 5,
        6144: 2,
        8192: 5,
        10240: 4,
        14336: 8,
        28672: 6,
        57344: 4
    },
    {  # tokens: [257:320]
        3072: 7,
        4096: 5,
        5120: 2,
        6144: 5,
        8192: 4,
        10240: 1,
        14336: 3,
        28672: 3,
        57344: 4
    },
    {  # tokens: [321:384]
        3072: 3,
        4096: 2,
        5120: 5,
        6144: 3,
        8192: 1,
        10240: 8,
        14336: 3,
        28672: 4,
        57344: 3
    },
    {  # tokens: [385:448]
        3072: 5,
        4096: 7,
        5120: 3,
        6144: 5,
        8192: 7,
        10240: 3,
        14336: 1,
        28672: 1,
        57344: 3
    },
    {  # tokens: [449:512]
        3072: 2,
        4096: 5,
        5120: 4,
        6144: 1,
        8192: 5,
        10240: 2,
        14336: 6,
        28672: 4,
        57344: 1
    },
    {  # tokens: [513:576]
        3072: 2,
        4096: 3,
        5120: 1,
        6144: 1,
        8192: 3,
        10240: 3,
        14336: 3,
        28672: 1,
        57344: 1
    },
    {  # tokens: [577:640]
        3072: 5,
        4096: 4,
        5120: 1,
        6144: 4,
        8192: 2,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [641:704]
        3072: 3,
        4096: 1,
        5120: 2,
        6144: 2,
        8192: 1,
        10240: 2,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [705:768]
        3072: 3,
        4096: 1,
        5120: 3,
        6144: 2,
        8192: 1,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    }
]


class QuantLlmLinearWeight(Tensor):
    _implements = classmethod(_implements)

    @staticmethod
    def new(cls, fpx_data: Tensor, scale: Tensor, ebits: int, mbits: int):
        assert fpx_data.ndim == 2
        assert fpx_data.dtype == torch.uint8
        shape = (fpx_data.shape[0], fpx_data.shape[1] // (1 + ebits + mbits) * 8)

        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=fpx_data.device,
            requires_grad=False,
        )

    def init(self, fpx_data: Tensor, scale: Tensor, ebits: int, mbits: int):
        self.fpx_data = fpx_data
        self.scale = scale
        self.ebits = ebits
        self.mbits = mbits

    def __tensor_flatten__(self):
        return ["fpx_data", "scale"], [self.ebits, self.mbits]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride):
        return cls(tensor_data_dict["fpx_data"], tensor_data_dict["scale"], *tensor_attributes)

    @classmethod
    def from_float(cls, input_float: Tensor, ebits: int, mbits: int):
        fpx_data, scale = _to_scaled_tc_fpx(input_float, ebits, mbits)
        return cls(fpx_data, scale, ebits, mbits)

    def dequantize(self, output_dtype=None):
        output_dtype = output_dtype or torch.get_default_dtype()
        return _from_tc_fpx(self.fpx_data, self.ebits, self.mbits) * self.scale.view(-1, 1)

    def repr(self):
        dtype = f"fp{1 + self.ebits + self.mbits}_e{self.ebits}m{self.mbits}"
        return (
            f"{self.__class__.name}(dtype={dtype}, shape={self.shape}, "
            f"device={self.device}, requires_grad={self.requires_grad})"
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.fpx_data),
            fn(self.scale),
            self.ebits,
            self.mbits,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(f"{cls.name} dispatch: attempting to run {func}, this is not supported")


@QuantLlmLinearWeight._implements(torch.nn.functional.linear)
def _(*args, **kwargs):
    act, weight, bias = args
    assert isinstance(weight, QuantLlmLinearWeight)

    out_dim, in_dim = weight.shape
    act_reshaped = act.view(-1, in_dim).half()

    # https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
    bsize = act_reshaped.shape[0]
    splitK = _SPLIT_K_MAP[(bsize - 1) // 64].get(out_dim, 1) if bsize <= 768 else 1

    out = quant_llm_linear(
        weight.ebits,
        weight.mbits,
        act_reshaped,
        weight.fpx_data,
        weight.scale,
        splitK=splitK,
    )

    if bias is not None:
        out += bias

    return out.view(*act.shape[:-1], out_dim).to(act.dtype)


@QuantLlmLinearWeight._implements(torch.ops.aten.detach.default)
def _(func, *args, **kwargs):
    return return_and_correct_aliasing(func, args, kwargs, args[0]._apply_fn_to_data(torch.detach))


class QuantLlmLinear(nn.Module):
    """Quant-LLM Linear layer as described in https://arxiv.org/pdf/2401.14112.
    """

    def __init__(
        self,
        ebits: int,
        mbits: int,
        weight: Tensor,
        scales: Tensor,
        bias: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] // (1 + ebits + mbits) * 8
        self.ebits = ebits
        self.mbits = mbits

    def forward(self, x: Tensor) -> Tensor:
        splitK = self.get_split_k(math.prod(x.shape[:-1]), self.out_features)
        out = quant_llm_linear(
            self.ebits,
            self.mbits,
            x.view(-1, self.in_features).half(),
            self.weight,
            self.scales,
            splitK=splitK,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.view(*x.shape[:-1], self.out_features).to(x.dtype)

    @staticmethod
    def get_split_k(bsize: int, out_dim: int) -> int:
        # https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
        return _SPLIT_K_MAP[(bsize - 1) // 64].get(out_dim, 1) if bsize <= 768 else 1

    @classmethod
    def from_float(cls, linear: nn.Linear, ebits: int, mbits: int):
        assert (linear.in_features % 64 == 0) and (linear.out_features % 256 == 0)

        fpx_weight, scale = _to_scaled_tc_fpx(linear.weight.detach(), ebits, mbits)
        bias = linear.bias.detach().half() if linear.bias is not None else None
        return cls(ebits, mbits, fpx_weight, scale, bias)

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}'
            f', out_features={self.out_features}'
            f', bias={self.bias is not None}'
            f', ebits={self.ebits}'
            f', mbits={self.mbits}'
        )


def convert_quant_llm(
    model: nn.Module,
    ebits: int,
    mbits: int,
    skip_fqn_list: Optional[List[str]] = None,
    cur_fqn: str = "",
) -> None:
    for name, child in model.named_children():
        new_fqn = name if cur_fqn == "" else f"{cur_fqn}.{name}"

        if ((skip_fqn_list is None) or (new_fqn not in skip_fqn_list)) and (isinstance(child, nn.Linear)):
            if (child.in_features % 64 == 0) and (child.out_features % 256 == 0):
                new_child = QuantLlmLinear.from_float(child, ebits, mbits)
                setattr(model, name, new_child)
        else:
            convert_quant_llm(child, ebits, mbits, skip_fqn_list, new_fqn)


def convert_fp6_llm(model: nn.Module, skip_fqn_list: Optional[List[str]] = None, cur_fqn: str = "") -> None:
    return convert_quant_llm(model, 3, 2, skip_fqn_list=skip_fqn_list, cur_fqn=cur_fqn)
