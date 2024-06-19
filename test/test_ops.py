import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.optests import opcheck
from torchao.utils import is_fbcode
from torchao.prototype.fp6_llm.fp6_llm import _from_tc_fpx
import pytest

if is_fbcode():
    pytest.skip("Skipping the test in fbcode since we don't have TARGET file for kernels")

try:
    import torchao.ops
except RuntimeError:
    pytest.skip("torchao.ops not available")


class TestOps(TestCase):
    def _create_fpx_inputs(self, ebits: int, mbits: int, BS: int, OC: int, IC: int, device):
        # Randomly initialize each byte
        nbits = 1 + ebits + mbits
        fpx_weight = torch.randint(256, (OC, IC // 8 * nbits), dtype=torch.uint8).view(torch.int32)
        scale = torch.rand(OC).half() + 0.5
        fp16_act = torch.rand(BS, IC).half() + 0.5
        return fpx_weight.to(device), scale.to(device), fp16_act.to(device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    def test_quant_llm_linear(self, ebits, mbits):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        fpx_weight, scale, fp16_act = self._create_fpx_inputs(ebits, mbits, BS, OC, IC, "cuda")

        # smoke test
        torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fpx_weight, scale, splitK)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.quant_llm_linear, (ebits, mbits, fp16_act, fpx_weight, scale, splitK), test_utils=test_utils)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("BS,OC,IC,splitK", [(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    def test_fp6_llm_linear_correctness(self, ebits, mbits, BS, OC, IC, splitK):
        # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/main/tests/python/kernel_test.py
        fpx_weight, scale, fp16_act = self._create_fpx_inputs(ebits, mbits, BS, OC, IC, "cuda")

        results_fpx = torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fpx_weight, scale, splitK)

        fp32_weight = _from_tc_fpx(fpx_weight.view(torch.uint8), ebits, mbits) * scale[:, None].float()
        fp16_weight = fp32_weight.half()
        results_fp16 = fp16_act @ fp16_weight.T

        error = (results_fpx - results_fp16).abs()
        relative_error = error / results_fp16.abs()
        assert relative_error.mean() < 1e-2


instantiate_parametrized_tests(TestOps)


if __name__ == "__main__":
    run_tests()
