#pragma once

#include "potrf_warppers.cuh"

namespace matrix_ops {
namespace cusolver {
template <typename T>
int cholesky(const common::CusolverDnHandle& handle, thrust::device_ptr<T> A,
             size_t n, size_t lda = 0) {
    return potrf<T>(handle, A, n, lda);
}
}  // namespace cusolver
}  // namespace matrix_ops
