#include "trsm_kernel.cuh"

namespace matrix_ops {
namespace okla {
template <>
__global__ void __launch_bounds__(128 * 4)
    trsm_kernel<float, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, 128, 128>(float alpha, const float* A,
                                                int lda, float* B, int ldb) {
    // 处理 RHS 的 4 列一组；blockIdx.x 选择哪一组
    constexpr int Nrhs_tile = 4;
    int row = threadIdx.x;          // 0..127
    int col_in_tile = threadIdx.y;  // 0..3
    int col = blockIdx.x * Nrhs_tile + col_in_tile;
    if (col >= 128) return;

    __shared__ float inv_diag[128];
    __shared__ float x_pivot[Nrhs_tile];

    // 预计算对角倒数（仅 col_in_tile==0 的线程做，避免重复）
    if (col_in_tile == 0) {
        inv_diag[row] = 1.0f / A[row + row * lda];  // col-major
    }
    __syncthreads();

    // Bij 放寄存器里做 in-place 更新；先乘 alpha（BLAS TRSM 语义中会用到
    // alpha）
    float Bij = alpha * B[row + col * ldb];

    // 前代入：k 从 0..127
    for (int k = 0; k < 128; ++k) {
        // pivot 行线程计算 x_k（本列）
        if (row == k) {
            float xk = Bij * inv_diag[k];
            x_pivot[col_in_tile] = xk;  // 广播给同列 tile 内的其他行
            Bij = xk;                   // B[k,col] 变成解
        }
        __syncthreads();

        // 用 x_k 更新下面的行
        if (row > k) {
            float Lik = A[row + k * lda];  // L(row,k)
            Bij -= Lik * x_pivot[col_in_tile];
        }
        __syncthreads();
    }

    // 写回（覆盖写回 B）
    B[row + col * ldb] = Bij;
}

template <>
__global__ void __launch_bounds__(256 * 4)
    trsm_kernel<float, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, 256, 256>(float alpha, const float* A,
                                                int lda, float* B, int ldb) {
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int Nrhs_tile = 4;

    int row = threadIdx.x;          // 0..255
    int col_in_tile = threadIdx.y;  // 0..3
    int col = blockIdx.x * Nrhs_tile + col_in_tile;
    if (col >= N) return;

    __shared__ float inv_diag[M];
    __shared__ float x_pivot[Nrhs_tile];

    // 预计算对角倒数：每行只算一次（tile 内 col_in_tile==0 的线程负责）
    if (col_in_tile == 0) {
        inv_diag[row] = 1.0f / A[row + row * lda];  // col-major
    }
    __syncthreads();

    // B(row, col) 放寄存器，in-place 更新；先乘 alpha
    float Bij = alpha * B[row + col * ldb];

// 前代入：k = 0..255
#pragma unroll 1
    for (int k = 0; k < M; ++k) {
        if (row == k) {
            float xk = Bij * inv_diag[k];
            x_pivot[col_in_tile] = xk;  // 广播给同 tile 的其他行
            Bij = xk;                   // 写回解到寄存器
        }
        __syncthreads();

        if (row > k) {
            float Lik = A[row + k * lda];  // L(row,k)
            Bij -= Lik * x_pivot[col_in_tile];
        }
        __syncthreads();
    }

    B[row + col * ldb] = Bij;
}
}  // namespace okla
}  // namespace matrix_ops
