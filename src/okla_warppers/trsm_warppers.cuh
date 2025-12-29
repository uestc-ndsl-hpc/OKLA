#include <thrust/device_vector.h>

#include <cstddef>
#include <type_traits>

#include "../common/log.h"
#include "../cusolver_warppers/trsm_warpper.cuh"

namespace matrix_ops {
namespace okla {

constexpr int max_block_num = 128;

// here max nrhs is gridDim.x * max_block_num
__global__ void trsm_kernel_f_128_n_left_lower_nt_nu(
    size_t nrhs, float alpha, const float* __restrict__ A, float* __restrict__ B, size_t lda,
    size_t ldb) {
    if (blockIdx.x >= nrhs) return;

    auto tx = threadIdx.x;
    constexpr int m = 128;

    __shared__ float shared_diag[m];
    if (tx < 32) {
#pragma unroll
        for (auto i = 4 * tx; i < 4 * tx + 4; i++) {
            shared_diag[i] = 1.0f / A[i * lda + i];
        }
    }
    __syncthreads();

    auto block_num = (nrhs + gridDim.x - 1) / gridDim.x;
    float* block_B = B + blockIdx.x * block_num * ldb;
    __shared__ float shared_X[max_block_num];

    for (auto row = 0; row < m; row++) {
        // solve X
        for (auto col = 0; col < block_num; col += blockDim.x) {
            if (col + tx < block_num) {
                shared_X[col + tx] =
                    block_B[row + (col + tx) * ldb] * shared_diag[row];
                block_B[row + (col + tx) * ldb] = shared_X[col + tx];
            }
        }
        __syncthreads();

        // update B
        for (auto col = 0; col < block_num; col++) {
            for (auto row_update = row + 1; row_update < m;
                 row_update += blockDim.x) {
                if (row_update + tx < m) {
                    block_B[row_update + tx + col * ldb] -=
                        shared_X[col] * A[row * lda + row_update + tx];
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
int trsm(const common::CublasHandle& handle, cublasSideMode_t side,
         cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
         size_t m, size_t n, T alpha, thrust::device_ptr<T> A,
         thrust::device_ptr<T> B, size_t lda = 0, size_t ldb = 0,
         size_t min_m = 128) {
    if (side != CUBLAS_SIDE_LEFT || uplo != CUBLAS_FILL_MODE_LOWER ||
        trans != CUBLAS_OP_N || diag != CUBLAS_DIAG_NON_UNIT) {
        util::Logger::println(
            "unsupported side, uplo, trans, diag = {} {} {} {}",
            util::cublasSideModeToString(side),
            util::cublasFillModeToString(uplo),
            util::cublasOperationToString(trans),
            util::cublasDiagTypeToString(diag));
        return -1;
    }
    if (lda == 0) {
        lda = (side == CUBLAS_SIDE_LEFT) ? m : n;
    }
    if (ldb == 0) {
        ldb = m;
    }
    if (m < min_m) {
        util::Logger::println("m < min_m = {}, use cusolver", min_m);
        return matrix_ops::cusolver::trsm<T>(handle, side, uplo, trans, diag, m,
                                             n, alpha, A, B, lda, ldb);
    }
    if constexpr (!std::is_same_v<T, float>) {
        util::Logger::println("okla trsm only supports float, use cusolver");
        return matrix_ops::cusolver::trsm<T>(handle, side, uplo, trans, diag, m,
                                             n, alpha, A, B, lda, ldb);
    }
    if (m != 128) {
        util::Logger::println("okla trsm only supports m == 128, use cusolver");
        return matrix_ops::cusolver::trsm<T>(handle, side, uplo, trans, diag, m,
                                             n, alpha, A, B, lda, ldb);
    }
    if (n == 0) {
        return 0;
    }

    constexpr int threads = 32;
    constexpr int blocks = 256;
    trsm_kernel_f_128_n_left_lower_nt_nu<<<blocks, threads>>>(
        n, alpha, thrust::raw_pointer_cast(A), thrust::raw_pointer_cast(B), lda,
        ldb);

    return 0;
}

}  // namespace okla
}  // namespace matrix_ops
