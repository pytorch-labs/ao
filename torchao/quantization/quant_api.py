# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization APIs

Generally these APIs can be applied directly to any model
with Linear modules to obtain quantized linear ops. The intended
usage involves applying torch.compile to the model afterwards
both because primitives were designed based on the fusions that
come along with it and because that is how we access the intended quantized
and mixed GEMM kernels
"""

import torch
from .dynamic_quant import (
    DynamicallyPerAxisQuantizedLinear,
)
from .subclass import (
    QuantizedLinearWeightBase,
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    Int4WeightOnlyQuantizedLinearWeight,
)
from .weight_only import (
    WeightOnlyInt8QuantLinear,
)

__all__ = [
    "apply_weight_only_int8_quant",
    "apply_dynamic_quant",
    "change_linear_weights_to_int8_dqtensors",
    "change_linear_weights_to_int8_woqtensors",
    "change_linear_weights_to_int4_woqtensors",
    "swap_conv2d_1x1_to_linear",
    "Quantizer",
    "TwoStepQuantizer",
]

############################# Unified Quantization APIs ##############################
# API 1, single quantize call to create a quantized model with quantized state_dict
class Quantizer:
    def quantize(self, model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        pass


# API 2, flow that needs calibration or training
class TwoStepQuantizer:
    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        pass

    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        pass

############################# Unified Quantization APIs ##############################

def _replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=""
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model


def _is_linear(mod, *args):
    return (
        isinstance(mod, torch.nn.Linear) and
        hasattr(mod, "weight") and
        not isinstance(mod.weight, QuantizedLinearWeightBase)
    )

def _in_features_greater_than_16(mod, *args):
    return hasattr(mod, "in_features") and mod.in_features > 16

def apply_weight_only_int8_quant(model, filter_fn=None):
    """
    Applies weight-only symmetric per-channel int8 quantization to all linear layers
    in the given model using module swaps.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        WeightOnlyInt8QuantLinear.from_float,
        _is_linear if filter_fn is None else filter_fn,
    )


def apply_dynamic_quant(model, filter_fn=None):
    """
    Applies dynamic symmetric per-token activation and per-channel weight
    quantization to all linear layers in the given model using
    module swaps.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: DynamicallyPerAxisQuantizedLinear.from_float(mod),
        _is_linear if filter_fn is None else filter_fn,
    )


def _get_subclass_inserter(cls, **kwargs):
    def insert_subclass(lin):
        lin.weight = torch.nn.Parameter(
            cls.from_float(lin.weight, **kwargs), requires_grad=False
        )
        return lin

    return insert_subclass


def change_linear_weights_to_int8_dqtensors(model, filter_fn=None):
    """
    Converts all linear weight tensors to the `Int8DynamicallyQuantizedLinearWeight`
    Tensor subclass, effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    if filter_fn is None:
        filter_fn = (
            lambda *args:
            _is_linear(*args) and
            _in_features_greater_than_16(*args)
        )

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int8DynamicallyQuantizedLinearWeight),
        filter_fn
    )


def change_linear_weights_to_int8_woqtensors(model, filter_fn=None):
    """
    Converts all linear weight tensors to the
    `Int8WeightOnlyQuantizedLinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int8WeightOnlyQuantizedLinearWeight),
        _is_linear if filter_fn is None else filter_fn,
    )


def change_linear_weights_to_int4_woqtensors(model, **kwargs):
    """
    Converts all linear weight tensors to the
    `Int4WeightOnlyQuantizedLinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    filter_fn = kwargs.pop("filter_fn", _is_linear)

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int4WeightOnlyQuantizedLinearWeight, **kwargs),
        filter_fn,
    )

def swap_conv2d_1x1_to_linear(model, filter_fn=None):
    """
    Changes all conv2d 1x1 modules to equivalent linear modules so that they can then be quantized.
    """
    class PermuteSandwich(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *args):
            return self.mod(args[0].permute(0, 2, 3, 1)).permute(-0,3,1,2)


    def replace_conv2d_1x1(conv):
        assert conv.kernel_size == (1, 1)
        lin = torch.nn.Linear(conv.in_channels, conv.out_channels, bias=(conv.bias is None))
        lin.weight=torch.nn.Parameter(conv.weight.squeeze(-1,-2))
        lin.bias = conv.bias
        return PermuteSandwich(lin)

    if filter_fn is None:
        filter_fn=lambda mod, *args: isinstance(mod, torch.nn.Conv2d) and mod.kernel_size==(1,1)

    _replace_with_custom_fn_if_matches_filter(
        model,
        replace_conv2d_1x1,
        filter_fn=filter_fn
    )
