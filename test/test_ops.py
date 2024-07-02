import itertools
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.optests import opcheck
from torchao.utils import is_fbcode
from torchao.prototype.quant_llm import from_scaled_tc_fpx
import pytest
from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    groupwise_affine_dequantize_tensor_from_qparams,
    pack_tinygemm_scales_and_zeros,
    unpack_tinygemm_scales_and_zeros
)
import torchao.quantization

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
        fpx_weight = torch.randint(256, (OC, IC // 8 * nbits), dtype=torch.uint8)
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
    def test_quant_llm_linear_correctness(self, ebits, mbits, BS, OC, IC, splitK):
        # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/tests/python/kernel_test_fpx.py
        fpx_weight, scale, fp16_act = self._create_fpx_inputs(ebits, mbits, BS, OC, IC, "cuda")

        results_fpx = torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fpx_weight, scale, splitK)

        fp16_weight = from_scaled_tc_fpx(fpx_weight, ebits, mbits, scale).half()
        results_fp16 = fp16_act @ fp16_weight.T

        error = (results_fpx - results_fp16).abs().mean()
        gt = results_fp16.abs().mean()
        relative_error = error / gt
        assert relative_error < 1e-3


instantiate_parametrized_tests(TestOps)

## Tests for `unpack_int4_packed`
kTileSizeN = 8
kTileSizeK = 16

SHAPES = [
    (4096, 4096),
    # Llama 2 GEMM shapes
    (4096, 11008),
    (11008, 4096),
    # Llama 3 GEMM shapes
    (4096, 14336),
    (14336, 4096),
]
INNERKTILES = [2, 4, 8]
QGROUP_SIZES = [32, 64, 128, 256]
TEST_CONFIGS_UNPACK = list(itertools.product(SHAPES, INNERKTILES))
TEST_CONFIGS_DEQUANT = list(itertools.product(SHAPES, INNERKTILES, QGROUP_SIZES))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS_UNPACK, ids=str)
def test_unpack_tensor_core_tiled_layout_correctness(shape, innerKTiles):
    N, K = shape
    assert K % (innerKTiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, innerKTiles)
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed_w, innerKTiles)
    assert torch.allclose(t, unpacked)

# TODO: Fix "test_aot_dispatch_dynamic" test failure
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS_UNPACK , ids=str)
def test_unpack_tensor_core_tiled_layout_op(shape, innerKTiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    ]
    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, innerKTiles)

    opcheck(
        torch.ops.torchao.unpack_tensor_core_tiled_layout,
        (packed_w, innerKTiles),
        test_utils=test_utils,
    )

def dequant_ref(q, scales, zeros, group_size, nbits=4, dtype=torch.bfloat16):
    n, k = q.shape
    assert q.dtype == torch.int

    n_groups = k // group_size
    assert scales.shape[0] == n and scales.shape[1] == n_groups
    assert scales.shape == zeros.shape

    midpoint = 2 ** (nbits - 1)
    
    #Convert fron u4 -> s4 and upcast to bfloat16
    q = q.sub(midpoint).to(dtype)
    
    # Dequantize
    q = q.reshape(-1, group_size)
    dq = q * scales.reshape(-1, 1) + zeros.reshape(-1, 1)

    return dq.reshape(n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, innerKTiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_correctness(shape, innerKTiles, group_size):
    n, k = shape
    dtype = torch.bfloat16    
    
    # tinygemm params
    nTileSize = 8
    kTileSize = 16
    nTiles = n // nTileSize
    kTiles = k // (innerKTiles * kTileSize)
    numThreads = 32

    device = "cuda"

    t = torch.randn(n, k, dtype=dtype, device=device)
    scales, zeros = get_groupwise_affine_qparams(t, n_bit=4, groupsize=group_size, dtype=dtype)
    
    # Quantize
    q = groupwise_affine_quantize_tensor_from_qparams(
        t, scales, zeros, n_bit=4, groupsize=group_size
    )
    
    # Pack to tensor core layout
    packed = torch.ops.aten._convert_weight_to_int4pack(q, innerKTiles)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
    q_groups = k // group_size
    assert scales_and_zeros.shape == torch.Size([q_groups, n, 2])

    dq_ao = groupwise_affine_dequantize_tensor_from_qparams(
        q, scales, zeros, n_bit=4, groupsize=group_size
    )
    
    # Dequantize by passing in an identity matrix as the activation
    a_eye = torch.eye(k, device=device, dtype=dtype)
    dq_id = torch.ops.aten._weight_int4pack_mm(
        a_eye,
        packed,
        group_size,
        scales_and_zeros,
    ).t()
    
    # Actual operation to test
    dq_op = torchao.ops.dequantize_tensor_core_tiled_layout(packed, scales_and_zeros, group_size, innerKTiles)
        
    # Compare results
    diff_ao_id = (dq_id - dq_ao).abs().max()
    diff_op_id = (dq_op - dq_id).abs().max()
    diff_op_ao = (dq_op - dq_ao).abs().max()
    
    # There are slight numerical differences when dequantizing with an identity matrix
    # Since the `dequantize_int4` kernel relies on same underlying numerical conversions, this gives same
    # numerical differences when compared to the `groupwise_affine_dequantize`
    
    # Test that the `dequant` kernel gives same results as identity matrix-based dequant 
    assert diff_op_id == 0
    
    # Test that the `dequant` kernel gives same numerical diffs as the `groupwise_affine_dequantize` when compared against the identity matrix
    assert diff_op_ao == diff_ao_id
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, innerKTiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_op(shape, innerKTiles, group_size):
    n, k = shape
    device = "cuda"

    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
    packed_w = torch._convert_weight_to_int4pack(q, innerKTiles)
    q_groups = k // group_size
    scales = torch.randn(n, q_groups, dtype=torch.bfloat16, device=device)
    zeros = torch.randn_like(scales)
    scales_and_zeros = torchao.quantization.utils.pack_tinygemm_scales_and_zeros(scales, zeros)
    
    test_utils = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
    ]
    opcheck(
        torch.ops.torchao.dequantize_tensor_core_tiled_layout,
        (packed_w, scales_and_zeros, group_size, innerKTiles),
        test_utils=test_utils,
    )

if __name__ == "__main__":
    run_tests()
