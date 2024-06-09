import logging
import unittest

import torch
from torch import nn

from torchao.sparsity import apply_fake_sparsity, apply_sparse_semi_structured
from torchao.sparsity.prototype.dynamic_quant_sparse import Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    _get_subclass_inserter,
    _is_linear,
)
from torchao.utils import TORCH_VERSION_AFTER_2_3
from torch.testing._internal.common_utils import TestCase


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

class TestSemiStructuredSparse(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        dense_result = model(input)

        apply_sparse_semi_structured(model)
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)


class TestQuantSemiSparse(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quant_semi_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        dense_result = model(input)

        _replace_with_custom_fn_if_matches_filter(model, _get_subclass_inserter(Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight), _is_linear)
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
