#include <cstddef>

#include "../common/log.h"
#include "../cusolver_warppers/trsm_warpper.cuh"

namespace matrix_ops {
namespace okla {

template <typename T>
int trsm(const common::CublasHandle& handle, cublasSideMode_t side,
         cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
         size_t m, size_t n, T alpha, thrust::device_ptr<T> A,
         thrust::device_ptr<T> B, size_t lda = 0, size_t ldb = 0,
         size_t min_m = 128) {
    if (side != CUBLAS_SIDE_LEFT || uplo != CUBLAS_FILL_MODE_LOWER ||
        trans != CUBLAS_OP_N || diag != CUBLAS_DIAG_NON_UNIT) {
        util::Logger::println(
            "unsupported side, uplo, trans, diag = {} {} {} {}", side, uplo,
            trans, diag);
        return -1;
    }
    if (m < min_m) {
        util::Logger::println("m < min_m = {}, use cusolver", min_m);
        return matrix_ops::cusolver::trsm<T>(handle, side, uplo, trans, diag, m,
                                             n, alpha, A, B, lda, ldb);
    }

    

    return 0;
}

}  // namespace okla
}  // namespace matrix_ops
