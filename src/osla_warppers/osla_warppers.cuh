#pragma once

#include <cstddef>
#include "potrf_warppers.cuh"

namespace matrix_ops {
namespace osla {
template <typename T>
int cholesky(const common::CusolverDnHandle& handle, thrust::device_ptr<T> A,
             size_t n, size_t lda = 0, size_t nb = 8192, size_t b = 1024) {
    return potrf<T>(handle, A, n, lda, nb, b);
}
}  // namespace osla
}  // namespace matrix_ops