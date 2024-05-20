#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("fp16act_fp6weight_linear(Tensor _in_feats, Tensor _weights, Tensor _scales, int splitK) -> Tensor");
  m.def("prepack_fp6_weight(Tensor fp6_tensor) -> Tensor");
  m.def("fp16_to_fp6_original(Tensor fp16_tensor) -> Tensor");
  m.def("fp6_weight_dequant(Tensor fp6_tensor, Tensor fp16_scale) -> Tensor");

  m.def("to_fp6_unpacked(Tensor fp16_tensor) -> Tensor");
  m.def("to_fp6_packed(Tensor fp16_tensor) -> Tensor");
  m.def("from_fp6_unpacked(Tensor fp6_tensor, ScalarType dtype) -> Tensor");
  m.def("from_fp6_packed(Tensor fp6_tensor, ScalarType dtype) -> Tensor");
}
