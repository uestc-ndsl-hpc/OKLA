#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>

#include "../common/handle_warppers.h"
#include "../cusolver_warppers/potrf_warppers.cuh"
#include "../matrix_ops/matrix_ops.cuh"

namespace matrix_ops {
namespace osla {
template <typename T>
int potrf(const common::CusolverDnHandle& handle, thrust::device_ptr<T> A,
          size_t n, size_t lda = 0, size_t nb = 8192, size_t b = 64) {
    if (n <= nb) {
        return matrix_ops::cusolver::potrf<T>(handle, A, n, lda);
    }

    lda = lda == 0 ? n : lda;

    auto cublas_handle = common::CublasHandle();

    constexpr bool kEnableLogging = true;
    util::Logger::init_timer(kEnableLogging);

    // init stream for inner update operations
    cudaStream_t stream_cusolver;
    cudaStreamCreateWithFlags(&stream_cusolver, cudaStreamNonBlocking);
    cudaStream_t ori_handle_cusolver_stream;
    cusolverDnGetStream(handle, &ori_handle_cusolver_stream);
    cusolverDnSetStream(handle, stream_cusolver);
    
    cudaStream_t stream_cublas;
    cudaStreamCreateWithFlags(&stream_cublas, cudaStreamNonBlocking);
    cudaStream_t ori_handle_cublas_stream;
    cublasGetStream(cublas_handle, &ori_handle_cublas_stream);
    cublasSetStream(cublas_handle, stream_cublas);

    for (auto outer_index = 0; outer_index < n; outer_index += nb) {
        for (auto inner_index = outer_index;
             inner_index < outer_index + nb - 1 && inner_index < n;
             inner_index += b) {
            // panel factorization
            util::Logger::tic("cholesky");
            matrix_ops::cusolver::potrf(
                handle, A + inner_index * lda + inner_index, b, lda);
            util::Logger::toc("cholesky", b * b * b);
            if (inner_index + b >= n) break;

            // update the elements below the panel
            // ops = (n - inner_index - b) * b * b;
            auto msg = fmt::format("trsm m: {} n: {}", n - inner_index - b, b);
            util::Logger::tic(msg);
            if constexpr (std::is_same_v<T, float>) {
                float alpha = 1.0f;
                cublasStrsm(cublas_handle, CUBLAS_SIDE_RIGHT,
                            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                            CUBLAS_DIAG_NON_UNIT, n - inner_index - b, b,
                            &alpha,
                            (const float*)thrust::raw_pointer_cast(
                                A + inner_index * lda + inner_index),
                            lda,
                            thrust::raw_pointer_cast(A + (inner_index + b) +
                                                     inner_index * lda),
                            lda);
            } else if constexpr (std::is_same_v<T, double>) {
                double alpha = 1.0;
                cublasDtrsm(cublas_handle, CUBLAS_SIDE_RIGHT,
                            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                            CUBLAS_DIAG_NON_UNIT, n - inner_index - b, b,
                            &alpha,
                            (const double*)thrust::raw_pointer_cast(
                                A + inner_index * lda + inner_index),
                            lda,
                            thrust::raw_pointer_cast(A + (inner_index + b) +
                                                     inner_index * lda),
                            lda);
            }
            util::Logger::toc(msg, (n - inner_index - b) * b * b);
            if (inner_index + b >= outer_index + nb - 1) break;

            // update the following panel
            constexpr bool kUseGemmForPanelUpdate = true;
            // When logging is disabled, fuse the two GEMMs into one to reduce launches.
            if constexpr (kUseGemmForPanelUpdate) {
                const auto k = static_cast<size_t>(inner_index + b - outer_index);
                const auto m_total = static_cast<size_t>(n - inner_index - b);
                if (m_total > 0 && k > 0) {
                    constexpr T alpha = static_cast<T>(-1.0);
                    constexpr T beta = static_cast<T>(1.0);
                    // Fused update: Cblk(inner_index+b : n-1, inner_index+b : inner_index+2b-1)
                    // = beta*Cblk + alpha * W * X^T
                    // where W = A(inner_index+b : n-1, outer_index : inner_index+b-1)
                    //       X = A(inner_index+b : inner_index+2b-1, outer_index : inner_index+b-1)
                    util::Logger::tic("gemm panel update");
                    matrix_ops::gemm(
                        cublas_handle, m_total, b, k, alpha,
                        A + (inner_index + b) + outer_index * lda, lda, false,
                        A + (inner_index + b) + outer_index * lda, lda, true,
                        beta,
                        A + (inner_index + b) + (inner_index + b) * lda, lda);
                    util::Logger::toc("gemm panel update", m_total * k * b);
                }
            } else {
                auto msg_local = fmt::format(
                    "syrk n:{} k:{} kUseGemmForPanelUpdate: {}", b,
                    inner_index + b - outer_index, kUseGemmForPanelUpdate);
                util::Logger::tic(msg_local);
                if constexpr (kUseGemmForPanelUpdate) {
                    constexpr T alpha = static_cast<T>(-1.0);
                    constexpr T beta = static_cast<T>(1.0);
                    matrix_ops::gemm(
                        cublas_handle, b, b, inner_index + b - outer_index,
                        alpha, A + (inner_index + b) + outer_index * lda, lda,
                        false, A + (inner_index + b) + outer_index * lda, lda,
                        true, beta,
                        A + inner_index + b + (inner_index + b) * lda, lda);
                } else {
                    if constexpr (std::is_same_v<T, float>) {
                        float alpha = -1.0f;
                        float beta = 1.0f;
                        cublasSsyrk(
                            cublas_handle, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, b, inner_index + b - outer_index,
                            &alpha,
                            thrust::raw_pointer_cast(A + (inner_index + b) +
                                                     outer_index * lda),
                            lda, &beta,
                            thrust::raw_pointer_cast(
                                A + inner_index + b +
                                (inner_index + b) * lda),
                            lda);
                    } else if constexpr (std::is_same_v<T, double>) {
                        double alpha = -1.0;
                        double beta = 1.0;
                        cublasDsyrk(
                            cublas_handle, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, b, inner_index + b - outer_index,
                            &alpha,
                            thrust::raw_pointer_cast(A + (inner_index + b) +
                                                     outer_index * lda),
                            lda, &beta,
                            thrust::raw_pointer_cast(
                                A + inner_index + b +
                                (inner_index + b) * lda),
                            lda);
                    }
                }
                util::Logger::toc(msg_local,
                                  b * (inner_index + b - outer_index) * b);

                if (inner_index + 2 * b >= n) continue;

                util::Logger::tic("gemm");
                matrix_ops::gemm(
                    cublas_handle, n - inner_index - 2 * b, b,
                    inner_index - outer_index + b, (T)-1.0,
                    A + (inner_index + 2 * b) + outer_index * lda, lda, false,
                    A + (inner_index + b) + outer_index * lda, lda, true,
                    (T)1.0,
                    A + (inner_index + 2 * b) + (inner_index + b) * lda, lda);
                util::Logger::toc(
                    "gemm", 2 * (n - inner_index - 2 * b) * b *
                                 (inner_index - outer_index + b));
            }
        }

        if (outer_index + nb >= n) break;

        util::Logger::tic("syrk trailing");
        // update the trailing matrix outer
        if constexpr (std::is_same_v<T, float>) {
            float alpha = -1.0f;
            float beta = 1.0f;
            cublasSsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nb,
                        outer_index + nb, &alpha,
                        thrust::raw_pointer_cast(A + outer_index + nb), lda,
                        &beta,
                        thrust::raw_pointer_cast(A + outer_index + nb +
                                                 (outer_index + nb) * lda),
                        lda);
        } else if constexpr (std::is_same_v<T, double>) {
            double alpha = -1.0;
            double beta = 1.0;
            cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nb,
                        outer_index + nb, &alpha,
                        thrust::raw_pointer_cast(A + outer_index + nb), lda,
                        &beta,
                        thrust::raw_pointer_cast(A + outer_index + nb +
                                                 (outer_index + nb) * lda),
                        lda);
        }
        util::Logger::toc("syrk trailing", nb * (outer_index + nb) * nb);

        if (outer_index + 2 * nb >= n) continue;

        util::Logger::tic("gemm trailing");
        matrix_ops::gemm(cublas_handle, n - outer_index - 2 * nb, nb,
                         outer_index + nb, (T)-1.0, A + outer_index + 2 * nb,
                         lda, false, A + outer_index + nb, lda, true, (T)1.0,
                         A + outer_index + 2 * nb + (outer_index + nb) * lda,
                         lda);
        util::Logger::toc("gemm trailing", 2 * (n - outer_index - 2 * nb) * nb *
                                               (outer_index + nb));
    }

    return 0;
}
}  // namespace osla
}  // namespace matrix_ops
