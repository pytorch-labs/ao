#include <cuda_fp16.h>
#include <iostream>
#include <assert.h>
#include <cstring>


// inspired by __internal_float2half() and float2half() from "cuda_fp16.hpp"
__device__ __host__ static uint8_t fp16_to_fp6(const __half a) {
    uint16_t bits;
    std::memcpy(&bits, &a, sizeof(a));

    uint16_t remainder = 0u;
    uint16_t sign = bits >> 15u << 5u;
    bits &= 0x7FFFu;  // clear sign bit
    uint16_t result;

#ifndef __CUDA_ARCH__
    if (bits >= 0b11111'0000000000u)
        throw std::invalid_argument("Encounter +/-inf or NaN, which is not representable in FP6.");
    if (bits >= 0b10011'1110000000u)
        throw std::invalid_argument("FP6 overflow. FP6 cannot represent +/-inf.");
#endif

    if (bits >= 0b01101'0000000000u) {  // FP6 normal number
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

__device__ __host__ static uint8_t fp32_to_fp6_v1(float a) {
#ifndef __CUDA_ARCH__
    if (std::isnan(a) | std::isinf(a))
        throw std::invalid_argument("Encounter +/-inf or NaN, which is not representable in FP6.");
    if (std::abs(a) >= 30.0f)  // 2^4 * (1 + 0.5 + 0.25 + 0.125)
        throw std::invalid_argument("FP6 overflow. FP6 cannot represent +/-inf.");
#endif

    a *= 0x1p-124;
    uint32_t bits;
    std::memcpy(&bits, &a, sizeof(a));

    uint8_t sign = bits >> 31u << 5u;
    uint8_t exp_and_man = (bits >> 21u) & 0x1Fu;
    uint8_t result = sign | exp_and_man;

    // round to nearest even
    uint32_t remainder = bits << 11u;
    if ((remainder > 0x8000'0000u) || ((remainder == 0x8000'0000u) && ((result & 1u) == 1u))) {
        result += 1;
    }

    return result;
}

// inspired by __internal_float2half() and float2half() from "cuda_fp16.hpp"
__device__ __host__ static uint8_t fp32_to_fp6_v2(const float a) {
    uint32_t bits;
    std::memcpy(&bits, &a, sizeof(a));

    uint32_t remainder = 0u;
    uint32_t sign = bits >> 31u << 5u;
    bits &= 0x7FFF'FFFFu;  // clear sign bit
    uint32_t result;

    constexpr uint32_t EXP_BIAS_DIFF = 127u - 3u;

    // only checks for invalid values on CPU, since we can't throw exception in CUDA
#ifndef __CUDA_ARCH__
    // all exponent bits are 1s
    if (bits >= (255u << 23u))
        throw std::invalid_argument("Encounter +/-inf or NaN, which is not representable in FP6.");

    // FP6 overflow when FP32 value is more than (or equal to) half way above max FP6 value
    // max FP6 is E=111, M=11. add extra 1 to M to get half way above it.
    if (bits >= (((EXP_BIAS_DIFF + 7u) << 23u) | (0b111 << 20u)))
        throw std::invalid_argument("FP6 overflow. FP6 cannot represent +/-inf.");
#endif

    // min FP6 subnormal number is 2^(-2) * 2^(-2)

    if (bits >= ((EXP_BIAS_DIFF + 1u) << 23u)) {  // FP6 normal number (E>=001)
        remainder = bits << (1u + 8u + 2u);
        bits -= (EXP_BIAS_DIFF << 23u);           // update exponent
        result = sign | (bits >> 21u);
    } else if (bits > ((EXP_BIAS_DIFF - 2u) << 23u)) {     // FP6 subnormal number
        uint32_t exp = bits >> 23u;
        uint32_t man = bits & 0x7F'FFFFu;

        // to make subnormal FP6 from normal FP16
        // step 1: add implicit 1 to mantissa
        man |= 0x80'0000u;

        // step 2: shift mantissa right so that exponent value is equal to
        // FP6 subnormal exponent value, which is -2
        uint32_t shift = 127u - 2u - exp;
        remainder = man << (1u + 8u + 2u + shift);
        man >>= shift;
        result = sign | (man >> 21u);  // implicit E=000
    } else {            // FP6 underflow
        result = sign;  // implicit E=000 and M=00
    }

    // round to nearest even
    if ((remainder > 0x8000'0000u) || ((remainder == 0x8000'0000u) && ((result & 1u) == 1u))) {
        result += 1;
    }

    return result;
}

#define fp32_to_fp6 fp32_to_fp6_v1

// assume the lower 6 bits contain the data
__device__ __host__ static float fp6_to_fp32(const uint8_t a) {
    // we shift the bits so that sign, exponent, and mantissa bits are in their correct positions in FP32.
    // this also handles subnormal numbers correctly.
    // FP6:                                  SE EEMM
    // FP32: S000 00EE EMM0 0000 0000 0000 0000 0000
    uint32_t bits = a;
    uint32_t sign = bits >> 5u << 31u;
    uint32_t exp_and_man = (bits & 0x1Fu) << 21u;
    uint32_t result_bits = sign | exp_and_man;

    // the result will be off by the difference in exponent bias
    // FP6:  Ebias =   3
    // FP32: Ebias = 127
    // correction = 2^(127-3)
    // we can correct this by direct FP32 multiplication, which also handles subnormal numbers correctly.
    float result;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result * 0x1p124;
}

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchao {

__global__ void fp16_to_fp6_unpacked_kernel(const __half *fp16_ptr, uint8_t *fp6_ptr, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        fp6_ptr[tid] = fp16_to_fp6(fp16_ptr[tid]);
}

// this is useful for debugging
at::Tensor fp16_to_fp6_unpacked(at::Tensor fp16_tensor) {
    TORCH_CHECK(fp16_tensor.dtype() == torch::kFloat16);
    TORCH_CHECK(fp16_tensor.is_contiguous());
    TORCH_CHECK(fp16_tensor.is_cpu() || fp16_tensor.is_cuda());
    
    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp16_tensor.device());
    at::Tensor fp6_tensor = at::empty(fp16_tensor.sizes(), options);

    const __half *fp16_ptr = reinterpret_cast<__half*>(fp16_tensor.data_ptr<at::Half>());
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp16_tensor.numel();

    if (fp16_tensor.is_cpu()) {
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < n; i++)
            fp6_ptr[i] = fp16_to_fp6(fp16_ptr[i]);
    } else {
        constexpr int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        fp16_to_fp6_unpacked_kernel<<<grid_size, block_size>>>(fp16_ptr, fp6_ptr, n);
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

at::Tensor fp16_to_fp6_packed(at::Tensor fp16_tensor) {
    TORCH_CHECK(fp16_tensor.dtype() == torch::kFloat16);
    TORCH_CHECK(fp16_tensor.is_contiguous());
    TORCH_CHECK(fp16_tensor.is_cpu() || fp16_tensor.is_cuda());
    TORCH_CHECK(fp16_tensor.ndimension() == 2);

    int M = fp16_tensor.size(0);
    int N = fp16_tensor.size(1);
    TORCH_CHECK(N % 4 == 0, "Last dimension must be a multiple of 4, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp16_tensor.device());
    at::Tensor fp6_tensor = at::empty({M, N * 3 / 4}, options);

    const __half *fp16_ptr = reinterpret_cast<__half*>(fp16_tensor.data_ptr<at::Half>());
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp16_tensor.numel();

    if (fp16_tensor.is_cpu()) {
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
    } else {
        constexpr int block_size = 256;
        int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
        fp16_to_fp6_packed_kernel<<<grid_size, block_size>>>(fp16_ptr, fp6_ptr, n);
    }

    return fp6_tensor;
}

__global__ void fp32_to_fp6_packed_kernel(const float *fp32_ptr, uint8_t *fp6_ptr, int n) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < n) {
        uint8_t val0 = fp32_to_fp6(fp32_ptr[idx]);
        uint8_t val1 = fp32_to_fp6(fp32_ptr[idx + 1]);
        uint8_t val2 = fp32_to_fp6(fp32_ptr[idx + 2]);
        uint8_t val3 = fp32_to_fp6(fp32_ptr[idx + 3]);

        fp6_ptr[idx / 4 * 3]     = (val0 << 2) | (val1 >> 4);  // 0000 0011
        fp6_ptr[idx / 4 * 3 + 1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
        fp6_ptr[idx / 4 * 3 + 2] = (val2 << 6) | (val3);       // 2233 3333
    }
}

at::Tensor fp32_to_fp6_packed(at::Tensor fp32_tensor) {
    TORCH_CHECK(fp32_tensor.dtype() == torch::kFloat32);
    TORCH_CHECK(fp32_tensor.is_contiguous());
    TORCH_CHECK(fp32_tensor.is_cpu() || fp32_tensor.is_cuda());
    TORCH_CHECK(fp32_tensor.ndimension() == 2);

    int M = fp32_tensor.size(0);
    int N = fp32_tensor.size(1);
    TORCH_CHECK(N % 4 == 0, "Last dimension must be a multiple of 4, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp32_tensor.device());
    at::Tensor fp6_tensor = at::empty({M, N * 3 / 4}, options);

    const float *fp32_ptr = fp32_tensor.data_ptr<float>();
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp32_tensor.numel();

    if (fp32_tensor.is_cpu()) {
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < n; i += 4) {
            uint8_t val0 = fp32_to_fp6(fp32_ptr[i]);
            uint8_t val1 = fp32_to_fp6(fp32_ptr[i + 1]);
            uint8_t val2 = fp32_to_fp6(fp32_ptr[i + 2]);
            uint8_t val3 = fp32_to_fp6(fp32_ptr[i + 3]);

            int j = i / 4 * 3;
            fp6_ptr[j]     = (val0 << 2) | (val1 >> 4);  // 0000 0011
            fp6_ptr[j + 1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
            fp6_ptr[j + 2] = (val2 << 6) | (val3);       // 2233 3333
        }
    } else {
        constexpr int block_size = 256;
        int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
        fp32_to_fp6_packed_kernel<<<grid_size, block_size>>>(fp32_ptr, fp6_ptr, n);
    }

    return fp6_tensor;
}

__global__ void fp6_unpacked_to_fp32_kernel(const uint8_t *fp6_ptr, float *fp32_ptr, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        fp32_ptr[idx] = fp6_to_fp32(fp6_ptr[idx]);
}

at::Tensor fp6_unpacked_to_fp32(at::Tensor fp6_tensor) {
    TORCH_CHECK(fp6_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(fp6_tensor.is_contiguous());
    TORCH_CHECK(fp6_tensor.is_cpu() || fp6_tensor.is_cuda());

    at::TensorOptions options = at::TensorOptions().dtype(torch::kFloat32).device(fp6_tensor.device());
    at::Tensor fp32_tensor = at::empty(fp6_tensor.sizes(), options);

    const uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    float *fp32_ptr = fp32_tensor.data_ptr<float>();
    int n = fp6_tensor.numel();

    if (fp6_tensor.is_cpu()) {
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < n; i++)
            fp32_ptr[i] = fp6_to_fp32(fp6_ptr[i]);
    } else {
        constexpr int block_size = 256;
        int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
        fp6_unpacked_to_fp32_kernel<<<grid_size, block_size>>>(fp6_ptr, fp32_ptr, n);
    }

    return fp32_tensor;
}

__global__ void fp6_packed_to_fp32_kernel(const uint8_t *fp6_ptr, float *fp32_ptr, int n) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    if (idx < n) {
        uint8_t bits0 = fp6_ptr[idx];      // 0000 0011
        uint8_t bits1 = fp6_ptr[idx + 1];  // 1111 2222
        uint8_t bits2 = fp6_ptr[idx + 2];  // 2233 3333

        int j = idx / 3 * 4;
        fp32_ptr[j]     = fp6_to_fp32(bits0 >> 2);
        fp32_ptr[j + 1] = fp6_to_fp32(((bits0 & 0x3u) << 4) | (bits1 >> 4));
        fp32_ptr[j + 2] = fp6_to_fp32(((bits1 & 0xFu) << 2) | (bits2 >> 6));
        fp32_ptr[j + 3] = fp6_to_fp32(bits2 & 0x3Fu);
    }
}

at::Tensor fp6_packed_to_fp32(at::Tensor fp6_tensor) {
    TORCH_CHECK(fp6_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(fp6_tensor.is_contiguous());
    TORCH_CHECK(fp6_tensor.is_cpu() || fp6_tensor.is_cuda());
    TORCH_CHECK(fp6_tensor.ndimension() == 2);

    int M = fp6_tensor.size(0);
    int N = fp6_tensor.size(1);
    TORCH_CHECK(N % 3 == 0, "Last dimension must be a multiple of 3, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kFloat32).device(fp6_tensor.device());
    at::Tensor fp32_tensor = at::empty({M, N / 3 * 4}, options);

    const uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    float *fp32_ptr = fp32_tensor.data_ptr<float>();
    int n = fp6_tensor.numel();

    if (fp6_tensor.is_cpu()) {
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < n; i += 3) {
            uint8_t bits0 = fp6_ptr[i];      // 0000 0011
            uint8_t bits1 = fp6_ptr[i + 1];  // 1111 2222
            uint8_t bits2 = fp6_ptr[i + 2];  // 2233 3333

            int j = i / 3 * 4;
            fp32_ptr[j]     = fp6_to_fp32(bits0 >> 2);
            fp32_ptr[j + 1] = fp6_to_fp32(((bits0 & 0x3u) << 4) | (bits1 >> 4));
            fp32_ptr[j + 2] = fp6_to_fp32(((bits1 & 0xFu) << 2) | (bits2 >> 6));
            fp32_ptr[j + 3] = fp6_to_fp32(bits2 & 0x3Fu);
        }
    } else {
        constexpr int block_size = 256;
        int grid_size = (n + block_size * 3 - 1) / (block_size * 3);
        fp6_packed_to_fp32_kernel<<<grid_size, block_size>>>(fp6_ptr, fp32_ptr, n);
    }

    return fp32_tensor;
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::fp16_to_fp6_unpacked", &fp16_to_fp6_unpacked);
  m.impl("torchao::fp16_to_fp6_packed", &fp16_to_fp6_packed);
  m.impl("torchao::fp32_to_fp6_packed", &fp32_to_fp6_packed);
  m.impl("torchao::fp6_unpacked_to_fp32", &fp6_unpacked_to_fp32);
  m.impl("torchao::fp6_packed_to_fp32", &fp6_packed_to_fp32);
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::fp16_to_fp6_unpacked", &fp16_to_fp6_unpacked);
  m.impl("torchao::fp16_to_fp6_packed", &fp16_to_fp6_packed);
  m.impl("torchao::fp32_to_fp6_packed", &fp32_to_fp6_packed);
  m.impl("torchao::fp6_unpacked_to_fp32", &fp6_unpacked_to_fp32);
  m.impl("torchao::fp6_packed_to_fp32", &fp6_packed_to_fp32);
}

}
