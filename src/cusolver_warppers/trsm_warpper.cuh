#pragma once

#include <cublas_v2.h>
#include <thrust/device_ptr.h>

#include <cstddef>

#include "../common/handle_warppers.h"

namespace matrix_ops {
namespace cusolver {
template <typename T>
int trsm(const common::CublasHandle& handle, cublasSideMode_t side,
         cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
         size_t m, size_t n, T alpha, thrust::device_ptr<T> A,
         thrust::device_ptr<T> B, size_t lda = 0, size_t ldb = 0) {
    if (lda == 0) {
        lda = (side == CUBLAS_SIDE_LEFT) ? m : n;
    }
    if (ldb == 0) {
        ldb = m;
    }

    if constexpr (std::is_same_v<T, float>) {
        return cublasStrsm(handle, side, uplo, trans, diag, static_cast<int>(m),
                           static_cast<int>(n), &alpha, A.get(),
                           static_cast<int>(lda), B.get(),
                           static_cast<int>(ldb));
    } else if constexpr (std::is_same_v<T, double>) {
        return cublasDtrsm(handle, side, uplo, trans, diag, static_cast<int>(m),
                           static_cast<int>(n), &alpha, A.get(),
                           static_cast<int>(lda), B.get(),
                           static_cast<int>(ldb));
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "cublas trsm only supports float and double");
    }
}
}  // namespace cusolver
}  // namespace matrix_ops
