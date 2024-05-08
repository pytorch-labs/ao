import torch
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
from torch.testing._internal.optests import opcheck
import torchao
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4
import unittest


# torch.testing._internal.optests.generate_tests.OpCheckError: opcheck(op, ...):
# test_faketensor failed with module 'torch' has no attribute '_custom_ops' (scroll up for stack trace)
@unittest.skipIf(IS_FBCODE, "Skipping the test in fbcode since we don't have TARGET file for kernels")
class TestOps(TestCase):
    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        # Adjust the threshold upward a bit with the intent of creating
        # at least one box that exceeds (barely) the threshold and so
        # should be suppressed.
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        iou_thresh += 1e-5
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.3 or lower")
    def test_nms(self):
        iou = 0.2
        boxes, scores = self._create_tensors_with_iou(1000, iou)
        boxes = boxes.cuda()
        scores = scores.cuda()

        # smoke test
        _ = torchao.ops.nms(boxes, scores, iou)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.nms, (boxes, scores, iou), test_utils=test_utils)

    def test_prepack_fp6_weight(self):
        OC = 256
        IC = 256
        fp6_weight = torch.randint(4294967295, (OC, IC // 16 * 3)).to(torch.int)

        # smoke test
        torchao.ops.prepack_fp6_weight(fp6_weight)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.prepack_fp6_weight, (fp6_weight,), test_utils=test_utils)

    def test_fake_fp6_to_fp6(self):
        OC = 256
        IC = 256

        # in this fp6, we use 3 bits for exponent and 2 bits for mantissa
        # also, we don't have nan/inf
        fp6_absmax = 28.0  # 2 ** (0b111 - 0b011) * (1 + 0.5 + 0.25), where E=111, M=11
        fp6_absmin = 0.0625  # 2 ** (-0b010) * 0.25, where E=000, M=01 (subnormal number)
        fake_fp6_weight = torch.randn((OC, IC), dtype=torch.float16)
        fake_fp6_weight.clip_(-fp6_absmax, fp6_absmax)
        fake_fp6_weight[fake_fp6_weight.abs() < fp6_absmin] = 0

        # smoke test
        torchao.ops.fake_fp6_to_fp6(fake_fp6_weight)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fake_fp6_to_fp6, (fake_fp6_weight,), test_utils=test_utils)


if __name__ == "__main__":
    unittest.main()
