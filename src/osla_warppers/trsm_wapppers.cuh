#include "../cusolver_warppers/trsm_warpper.cuh"
#include "../matrix_ops/matrix_ops.cuh"

namespace matrix_ops {
namespace osla {
template <typename T>
int trsm(const common::CublasHandle& handle, cublasSideMode_t side,
         cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
         size_t m, size_t n, T alpha, thrust::device_ptr<T> A,
         thrust::device_ptr<T> B, size_t lda = 0, size_t ldb = 0,
         size_t nb = 8192, size_t b = 64) {
    if (n <= nb) {
        return matrix_ops::cusolver::trsm<T>(handle, side, uplo, trans, diag, m,
                                             n, alpha, A, B, lda, ldb);
    }
    lda = lda == 0 ? (side == CUBLAS_SIDE_LEFT) ? m : n : lda;
    ldb = ldb == 0 ? m : ldb;

    // for outer in range(0, n, nb):
    for (auto outer_index = 0; outer_index < n; outer_index += nb) {
        auto outer_end = std::min(outer_index + nb, n);
        auto inner_index = outer_index;
        while (inner_index < outer_end) {
            auto inner_end = std::min(inner_index + b, outer_end);
            auto bs = inner_end - inner_index;

            matrix_ops::cusolver::trsm(
                handle, side, uplo, trans, diag, bs, n, alpha,
                A + inner_index * lda + inner_index, B + inner_index, lda, ldb);

            auto next_start = inner_end;
            if (next_start < outer_end) {
                auto next_end = std::min(next_start + bs, outer_end);
                matrix_ops::gemm(
                    handle, next_end - next_start, n, inner_end - outer_index,
                    (T)-1.0, A + next_start + outer_index * lda, lda,
                    B + outer_index, ldb, (T)1.0, B + next_start, ldb);
            }
            inner_index = inner_end;
        }
        if (outer_end < n) {
            matrix_ops::gemm(handle, n - outer_end, n, outer_end - outer_index,
                             (T)-1.0, A + outer_end + outer_index * lda, lda,
                             B + outer_index, ldb, (T)1.0, B + outer_end, ldb);
        }
    }
    return 0;
}
}  // namespace osla
}  // namespace matrix_ops