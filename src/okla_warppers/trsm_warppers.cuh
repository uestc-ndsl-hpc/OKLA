#include <cstddef>
#include <type_traits>

#include "../cusolver_warppers/trsm_warpper.cuh"
#include "trsm_kernel.cuh"

namespace matrix_ops {
namespace okla {

template <typename T, cublasSideMode_t SideMode, cublasFillMode_t FillMode,
          cublasOperation_t Operation, cublasDiagType_t Diag, int M, int Nrhs>
cudaError_t launch_trsm_kernel(dim3 grid, dim3 block, cudaStream_t stream,
                               T alpha, const T* A, int lda, T* B, int ldb,
                               size_t shmem = 0) {
    void* args[] = {(void*)&alpha, (void*)&A, (void*)&lda, (void*)&B,
                    (void*)&ldb};

    return cudaLaunchKernel(
        (const void*)
            trsm_kernel<T, SideMode, FillMode, Operation, Diag, M, Nrhs>,
        grid, block, args, shmem, stream);
}

template <typename T>
int trsm_dispatch_okla(cudaStream_t stream, cublasSideMode_t side,
                       cublasFillMode_t uplo, cublasOperation_t trans,
                       cublasDiagType_t diag, int m, int n, T alpha, const T* A,
                       int lda, T* B, int ldb) {
    if (!(side == CUBLAS_SIDE_LEFT && uplo == CUBLAS_FILL_MODE_LOWER &&
          trans == CUBLAS_OP_N && diag == CUBLAS_DIAG_NON_UNIT)) {
        return -1;
    }

    auto launch = [&](auto M_c, auto N_c) -> cudaError_t {
        constexpr int Mv = decltype(M_c)::value;
        constexpr int Nv = decltype(N_c)::value;
        constexpr int Nrhs_tile = 4;
        dim3 block(128, Nrhs_tile);
        dim3 grid((Nv + Nrhs_tile - 1) / Nrhs_tile, 1);
        return launch_trsm_kernel<T, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                  CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Mv, Nv>(
            grid, block, stream, alpha, A, lda, B, ldb);
    };

    if (m == 128 && n == 128) {
        return launch(std::integral_constant<int, 128>{},
                      std::integral_constant<int, 128>{});
    }

    if (m == 256 && n == 256) {
        return launch(std::integral_constant<int, 256>{},
                      std::integral_constant<int, 256>{});
    }

    return -1;
}

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

    cudaStream_t stream = nullptr;
    cublasGetStream(handle, &stream);
    return trsm_dispatch_okla<T>(stream, side, uplo, trans, diag,
                                 static_cast<int>(m), static_cast<int>(n),
                                 alpha, A.get(), static_cast<int>(lda), B.get(),
                                 static_cast<int>(ldb));
}
}  // namespace okla
}  // namespace matrix_ops
