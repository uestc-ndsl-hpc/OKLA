#pragma once

#include <cusolverDn.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>

#include "../common/handle_warppers.h"

namespace matrix_ops {
namespace cusolver {
template <typename T>
int potrf(const common::CusolverDnHandle& handle, thrust::device_ptr<T> A,
          size_t n, size_t lda = 0) {
    auto data_type = CUDA_R_32F;
    auto compute_type = CUDA_R_32F;

    if constexpr (std::is_same_v<T, double>) {
        data_type = CUDA_R_64F;
        compute_type = CUDA_R_64F;
    } else if constexpr (std::is_same_v<T, float>) {
        data_type = CUDA_R_32F;
        compute_type = CUDA_R_32F;
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "cusolver potrf only supports float and double");
    }

    cusolverDnParams_t params = NULL;
    cusolverDnCreateParams(&params);

    // pre allocate workspace
    size_t size_d = 0;
    size_t size_h = 0;
    lda = lda == 0 ? n : lda;
    auto status = cusolverDnXpotrf_bufferSize(
        handle, params, CUBLAS_FILL_MODE_LOWER, n, data_type, A.get(), lda,
        compute_type, &size_d, &size_h);
    auto d_work = thrust::device_vector<char>(size_d);
    auto h_work = thrust::host_vector<char>(size_h);

    // call potrf
    thrust::device_vector<int> info_d(1);
    nvtxRangePushA("cholesky");
    cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, n, data_type,
                     A.get(), lda, compute_type, d_work.data().get(), size_d,
                     h_work.data(), size_h, info_d.data().get());
    cudaStream_t s{};
    cusolverDnGetStream(handle, &s);  // 若你自己 set 了 stream，就直接用那个
    cudaStreamSynchronize(s);         // 确保 NVTX 范围里包含真正的 GPU 执行
    nvtxRangePop();
    thrust::host_vector<int> info_h = info_d;
    return info_h[0];
}
}  // namespace cusolver
}  // namespace matrix_ops