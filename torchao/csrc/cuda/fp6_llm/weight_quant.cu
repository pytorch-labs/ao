//    Copyright 2024 FP6-LLM authors
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
// 
// This file is adapted from https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/utils/weight_quant.h
// and https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/utils/weight_dequant.h

#include <cuda_fp16.h>
#include <iostream>
#include <assert.h>
#include <cstring>


// inspired by __internal_float2half() and float2half() from "cuda_fp16.hpp"
__device__ __host__ uint8_t fp16_to_fp6(const __half a) {
    uint16_t bits;
    std::memcpy(&bits, &a, sizeof(a));

    uint16_t remainder = 0u;
    uint16_t sign = bits >> 15u << 5u;
    bits &= 0x7FFFu;  // clear sign bit
    uint16_t result;

    if (bits >= 0b11111'0000000000u) {
#ifndef __CUDA_ARCH__
        throw std::invalid_argument("Encounter +/-inf or NaN, which is not representable in FP6.");
#endif
    } else if (bits >= 0b10011'1110000000u) {
#ifndef __CUDA_ARCH__
        throw std::invalid_argument("FP6 overflow. FP6 cannot represent +/-inf.");
#endif
    } else if (bits >= 0b01101'0000000000u) {  // FP6 normal number
        remainder = bits << 8u;
        bits -= (0b01100u << 10u);  // update exponent
        result = sign | (bits >> 8u);
    } else if (bits >= 0b01010'0000000001u) {  // FP6 subnormal number
        uint16_t exp = bits >> 10u;
        uint16_t man = bits & 0x3FFu;
        uint16_t shift = 0b01111u - 0b011u + 1u + 8u - exp;
        man |= 0x400u;  // set implicit 1 to mantissa
        remainder = man << (16u - shift);
        man >>= shift;
        result = sign | man;
    } else {  // FP6 underflow
        result = sign;
    }

    // round to nearest even
    if ((remainder > 0x8000u) || ((remainder == 0x8000u) && ((result & 1u) == 1u))) {
        result += 1;
    }

    return result;
}

// assume the lower 6 bits contain the data
__device__ __host__ __half fp6_to_fp16(const uint8_t a) {
    // we shift the bits so that sign, exponent, and mantissa bits are in their
    // correct positions in FP16
    // FP6:              SE EEMM
    // FP16: S00E EEMM 0000 0000
    uint16_t bits = a;
    uint16_t sign = (a << 10u) & 0x8000u;
    uint16_t exp_and_man = (a & 0x1Fu) << 8u;
    uint16_t result_bits = sign | exp_and_man;

    // the result will be off by the difference in exponent bias
    // FP6:  Ebias =   011 = 2^3
    // FP16: Ebias = 01111 = 2^15
    // correction = 2^12 = 4096
    // we can correct this by direct FP16 multiplication
    __half result;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result * __float2half(4096.0f);
}

/*
 * Function to pack 4 fake quantized FP16 value into continuously stored 4 FP6 values.
 */
void cast_fp16_fp6(uint16_t* FP16x4, uint8_t* FP6x4)
{
    // Constants for FP6
    constexpr int exponent_nbits_fp6 = 3;
    constexpr int mantissa_nbits_fp6 = 2;
    constexpr int exp_bias_fp6 = (1 << (exponent_nbits_fp6 - 1)) - 1;
    // Constants for FP16
    constexpr int exponent_nbits_fp16 = 5;
    constexpr int mantissa_nbits_fp16 = 10;
    constexpr int exp_bias_fp16 = (1 << (exponent_nbits_fp16 - 1)) - 1;

    int fp6_temp[4];

    float absmin_nonzero_fp6 = 0.0625;
    // Note that we regard the exponent of '111' as a regular value rather than NaN or inf. This is
    // the same with that in qtorch.
    float absmax_fp6 = 28;

    for (int i = 0; i < 4; ++i) {
        uint16_t source = FP16x4[i];
        float fp6_value_abs = std::abs(__half2float(*((half*)(&source))));
        if ((fp6_value_abs != 0 && fp6_value_abs < absmin_nonzero_fp6) ||
            fp6_value_abs > absmax_fp6) {
            // TODO(zhen): a better way may be rounding it to the nearest FP6 value.
            throw std::invalid_argument("Input value out of range for FP6.");
        }

        // It is not safe to do shift operation on uint16_t. So we promote it to int.
        int source_promote = int(source);

        int sign_bit = (source_promote >> 15);
        // Extracting exponent represented in FP16. The sign mask 0x7FFF is '0111 1111 1111 1111'
        int exp_bit = (source_promote & 0x7FFF) >> mantissa_nbits_fp16;
        // Extracting mantissa represented in FP16
        int mant_bit = source_promote & ((1 << mantissa_nbits_fp16) - 1);

        int new_exp_bit;
        int new_mant_bit;

        if (exp_bit == 0) {
            // Subnormal FP16 number. Too small for FP6.
            new_exp_bit = 0;
            new_mant_bit = 0;
        } else {
            new_mant_bit = mant_bit >> (mantissa_nbits_fp16 - mantissa_nbits_fp6);
            new_exp_bit = exp_bit - exp_bias_fp16 + exp_bias_fp6;

            // Deal with subnormal FP6 values.
            int target_exp_val = exp_bit - exp_bias_fp16;
            int min_fp6_exp_val = -exp_bias_fp6 + 1;
            bool subnormal_fp6 = target_exp_val < min_fp6_exp_val;
            if (subnormal_fp6) {
                // TODO(zhen): add the rounding logic.
                new_exp_bit = 0;
                // The implicit 1 in the mantissa of FP16 is not present in subnormal FP6. Thus we
                // need to add it
                new_mant_bit = (new_mant_bit | (1 << mantissa_nbits_fp6)) >>
                               (min_fp6_exp_val - target_exp_val);
            }
        }

        fp6_temp[i] = (sign_bit << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                      (new_exp_bit << mantissa_nbits_fp6) | new_mant_bit;
    }
    // Pack the values
    FP6x4[0] = fp6_temp[0] << 2 | (fp6_temp[1] >> 4);
    FP6x4[1] = (fp6_temp[1] & 0x0F) << 4 | (fp6_temp[2] >> 2);
    FP6x4[2] = (fp6_temp[2] & 0x03) << 6 | fp6_temp[3];
}

/*
 *  Function to prepack FP16 weights into continuous FP6 values.
 *
 *  Parameters:
 *     weight_16bit: input weight in FP16, size M*K
 *     weight_6bit: output weight in packed FP6, continuously stored, size M*K*6/8
 *     M, K: the shape of the weight
 */
void weight_prepacking_fp16_to_fp6(uint16_t* weight_16bit,
                                   uint8_t* weight_6bit_packed,
                                   size_t M,
                                   size_t K)
{
    // Every four 16-bit elements are packed into three 6-bit values (4*6bit == 3*8bit).
    if (K * 6 % 8 != 0) { throw std::invalid_argument("(K * 6 % 8) should be 0"); }
    size_t K_fp6_packed = K * 6 / 8;
    // #pragma omp parallel for
    for (auto m = 0; m < M; m++) {
        uint8_t* ptr_6bit = weight_6bit_packed + m * K_fp6_packed;
        uint16_t* ptr_16bit = weight_16bit + m * K;
        for (auto k = 0; k < K; k += 4) {
            cast_fp16_fp6(ptr_16bit, ptr_6bit);
            ptr_16bit += 4;
            ptr_6bit += 3;
        }
    }
}

void DeQuantMatrix_FP6_To_FP16(half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale) {
    assert(M%64==0);                 // Currently, M must be a multiple of 64.
    assert(K%64==0);                 // Currently, K must be a multiple of 64.
    size_t TotalSizeInByte = M*K*6/8;
    //
    half* OutPTR = A_16bit_h;
    for(size_t i=0; i<TotalSizeInByte/3; i++) {    // 4 FP6 = 3 Bytes for each Loop
        unsigned char   B1  = A_6bit_h[i*3+0] & 0xfc;
                        B1  = (B1&0x80) | ((B1>>2)&0x1f);
        unsigned char   B2  = (A_6bit_h[i*3+0]<<6) | ((A_6bit_h[i*3+1]>>2)&0xfc);
                        B2  = (B2&0x80) | ((B2>>2)&0x1f);
        unsigned char   B3  = (A_6bit_h[i*3+1]<<4) | ((A_6bit_h[i*3+2]>>4)&0xfc);
                        B3  = (B3&0x80) | ((B3>>2)&0x1f);
        unsigned char   B4  = A_6bit_h[i*3+2]<<2;
                        B4  = (B4&0x80) | ((B4>>2)&0x1f);
        half            FP1, FP2, FP3, FP4;
        unsigned char   *PTR1, *PTR2, *PTR3, *PTR4;
        PTR1 = reinterpret_cast<unsigned char*>(&FP1);
        PTR2 = reinterpret_cast<unsigned char*>(&FP2);
        PTR3 = reinterpret_cast<unsigned char*>(&FP3);
        PTR4 = reinterpret_cast<unsigned char*>(&FP4);
        PTR1[0] = 0;    PTR1[1] = B1;   // small endian for X86 CPU
        PTR2[0] = 0;    PTR2[1] = B2;
        PTR3[0] = 0;    PTR3[1] = B3;
        PTR4[0] = 0;    PTR4[1] = B4;
        OutPTR[0] = __float2half_rn ( __half2float(FP1) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[1] = __float2half_rn ( __half2float(FP2) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[2] = __float2half_rn ( __half2float(FP3) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[3] = __float2half_rn ( __half2float(FP4) * 4096.0f * __half2float(scale[(4*i)/K]) );
        //
        OutPTR +=4;
    }
}


#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchao {

// https://github.com/microsoft/DeepSpeed/blob/0fc19b6a320cf8aa0a5f6c2b1fa310bae9a70d94/deepspeed/inference/v2/kernels/core_ops/cuda_linear/linear_kernels.cpp#L194
at::Tensor fp16_to_fp6_original_cpu(at::Tensor fp16_tensor)
{
    TORCH_CHECK(fp16_tensor.dim() == 2, "weight must be 2-dimensional");
    TORCH_CHECK(fp16_tensor.scalar_type() == torch::kFloat16, "weight must be FP16");
    TORCH_CHECK(fp16_tensor.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(fp16_tensor.device().type() == torch::kCPU, "weight must be on CPU");
    auto M = fp16_tensor.size(0);
    auto K = fp16_tensor.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4");

    // Pack weight from FP16 to FP6.
    auto options = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto packed_fp6_tensor = at::empty({M, K * 6 / 8}, options);
    uint8_t* packed_fp6_ptr = packed_fp6_tensor.data_ptr<uint8_t>();

    uint16_t* fake_fp6_ptr = reinterpret_cast<uint16_t*>(fp16_tensor.data_ptr<at::Half>());
    weight_prepacking_fp16_to_fp6(fake_fp6_ptr, packed_fp6_ptr, M, K);

    return packed_fp6_tensor;
}

/*
 * Dequant a FP6 matrix to a equivalent FP16 matrix using CPUs.
 * A useful tool to construct input matrices for the FP16 GEMM baseline.
 * [Input]
 *  fp6_tensor:  int  tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6  weights.
 *  fp16_scale:  half tensor of shape [OC];                 // for row-wise quantization.
 * [Output]
 *  fp16_tensor: half tensor of shape [OC, IC].     
 */
at::Tensor weight_matrix_dequant_cpu(at::Tensor fp6_tensor, at::Tensor fp16_scale) 
{
    int OC = fp6_tensor.size(0);
    TORCH_CHECK(fp6_tensor.size(1) % 3 == 0);
    int IC = fp6_tensor.size(1) / 3 * 16;
    TORCH_CHECK(fp16_scale.size(0) == OC);
    //
    auto fp6_tensor_ptr = reinterpret_cast<unsigned char*>(fp6_tensor.data_ptr<int>());
    auto fp16_scale_ptr = reinterpret_cast<half*>(fp16_scale.data_ptr<at::Half>());
    //
    auto options = at::TensorOptions().dtype(at::kHalf).device(fp16_scale.device());
    at::Tensor fp16_tensor = at::empty({OC, IC}, options);
    auto fp16_tensor_ptr = reinterpret_cast<half*>(fp16_tensor.data_ptr<at::Half>());
    //
    DeQuantMatrix_FP6_To_FP16(fp16_tensor_ptr, fp6_tensor_ptr, OC, IC, fp16_scale_ptr);
    //
    return fp16_tensor;
}

// this is used for debugging
at::Tensor fp16_to_fp6_unpacked_cpu(at::Tensor fp16_tensor) {
    TORCH_CHECK(fp16_tensor.dtype() == torch::kFloat16);
    TORCH_CHECK(fp16_tensor.is_contiguous());
    TORCH_CHECK(fp16_tensor.is_cpu());
    
    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp16_tensor.device());
    at::Tensor fp6_tensor = at::empty(fp16_tensor.sizes(), options);

    const __half *fp16_ptr = reinterpret_cast<__half*>(fp16_tensor.data_ptr<at::Half>());
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp16_tensor.numel();

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        fp6_ptr[i] = fp16_to_fp6(fp16_ptr[i]);
    }

    return fp6_tensor;
}

__global__ void fp16_to_fp6_unpacked_kernel(const __half *fp16_ptr, uint8_t *fp6_ptr, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        fp6_ptr[tid] = fp16_to_fp6(fp16_ptr[tid]);
    }
}

at::Tensor fp16_to_fp6_unpacked_cuda(at::Tensor fp16_tensor) {
    TORCH_CHECK(fp16_tensor.dtype() == torch::kFloat16);
    TORCH_CHECK(fp16_tensor.is_contiguous());
    TORCH_CHECK(fp16_tensor.is_cuda());
    
    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp16_tensor.device());
    at::Tensor fp6_tensor = at::empty(fp16_tensor.sizes(), options);

    const __half *fp16_ptr = reinterpret_cast<__half*>(fp16_tensor.data_ptr<at::Half>());
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp16_tensor.numel();

    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    fp16_to_fp6_unpacked_kernel<<<grid_size, block_size>>>(fp16_ptr, fp6_ptr, n);

    return fp6_tensor;
}

at::Tensor fp16_to_fp6_packed_cpu(at::Tensor fp16_tensor) {
    TORCH_CHECK(fp16_tensor.dtype() == torch::kFloat16);
    TORCH_CHECK(fp16_tensor.is_contiguous());
    TORCH_CHECK(fp16_tensor.is_cpu());
    TORCH_CHECK(fp16_tensor.ndimension() == 2);

    int M = fp16_tensor.size(0);
    int N = fp16_tensor.size(1);
    TORCH_CHECK(N % 4 == 0, "Last dimension must be a multiple of 4, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp16_tensor.device());
    at::Tensor fp6_tensor = at::empty({M, N * 3 / 4}, options);

    const __half *fp16_ptr = reinterpret_cast<__half*>(fp16_tensor.data_ptr<at::Half>());
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp16_tensor.numel();

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i += 4) {
        uint8_t val0 = fp16_to_fp6(fp16_ptr[i]);
        uint8_t val1 = fp16_to_fp6(fp16_ptr[i + 1]);
        uint8_t val2 = fp16_to_fp6(fp16_ptr[i + 2]);
        uint8_t val3 = fp16_to_fp6(fp16_ptr[i + 3]);

        int j = i / 4 * 3;
        fp6_ptr[j]     = (val0 << 2) | (val1 >> 4);  // 0000 0011
        fp6_ptr[j + 1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
        fp6_ptr[j + 2] = (val2 << 6) | (val3);       // 2233 3333
    }

    return fp6_tensor;
}

__global__ void fp16_to_fp6_packed_kernel(const __half *fp16_ptr, uint8_t *fp6_ptr, int n) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < n) {
        uint8_t val0 = fp16_to_fp6(fp16_ptr[idx]);
        uint8_t val1 = fp16_to_fp6(fp16_ptr[idx + 1]);
        uint8_t val2 = fp16_to_fp6(fp16_ptr[idx + 2]);
        uint8_t val3 = fp16_to_fp6(fp16_ptr[idx + 3]);

        fp6_ptr[idx / 4 * 3]     = (val0 << 2) | (val1 >> 4);  // 0000 0011
        fp6_ptr[idx / 4 * 3 + 1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
        fp6_ptr[idx / 4 * 3 + 2] = (val2 << 6) | (val3);       // 2233 3333
    }
}

at::Tensor fp16_to_fp6_packed_cuda(at::Tensor fp16_tensor) {
    TORCH_CHECK(fp16_tensor.dtype() == torch::kFloat16);
    TORCH_CHECK(fp16_tensor.is_contiguous());
    TORCH_CHECK(fp16_tensor.is_cuda());
    TORCH_CHECK(fp16_tensor.ndimension() == 2);

    int M = fp16_tensor.size(0);
    int N = fp16_tensor.size(1);
    TORCH_CHECK(N % 4 == 0, "Last dimension must be a multiple of 4, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp16_tensor.device());
    at::Tensor fp6_tensor = at::empty({M, N * 3 / 4}, options);

    const __half *fp16_ptr = reinterpret_cast<__half*>(fp16_tensor.data_ptr<at::Half>());
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp16_tensor.numel();

    constexpr int block_size = 256;
    int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
    fp16_to_fp6_packed_kernel<<<grid_size, block_size>>>(fp16_ptr, fp6_ptr, n);

    return fp6_tensor;
}

at::Tensor fp6_unpacked_to_fp16_cpu(at::Tensor fp6_tensor) {
    TORCH_CHECK(fp6_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(fp6_tensor.is_contiguous());
    TORCH_CHECK(fp6_tensor.is_cpu());

    at::TensorOptions options = at::TensorOptions().dtype(torch::kFloat16).device(fp6_tensor.device());
    at::Tensor fp16_tensor = at::empty(fp6_tensor.sizes(), options);

    const uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    __half *fp16_ptr = reinterpret_cast<__half *>(fp16_tensor.data_ptr<at::Half>());
    int n = fp6_tensor.numel();

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        fp16_ptr[i] = fp6_to_fp16(fp6_ptr[i]);
    }

    return fp16_tensor;
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::fp16_to_fp6_original", &fp16_to_fp6_original_cpu);
  m.impl("torchao::fp6_weight_dequant", &weight_matrix_dequant_cpu);
  m.impl("torchao::fp16_to_fp6_unpacked", &fp16_to_fp6_unpacked_cpu);
  m.impl("torchao::fp16_to_fp6_packed", &fp16_to_fp6_packed_cpu);
  m.impl("torchao::fp6_unpacked_to_fp16", &fp6_unpacked_to_fp16_cpu);
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::fp16_to_fp6_unpacked", &fp16_to_fp6_unpacked_cuda);
  m.impl("torchao::fp16_to_fp6_packed", &fp16_to_fp6_packed_cuda);
}

}
