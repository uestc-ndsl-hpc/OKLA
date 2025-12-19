#include <cublas_v2.h>
#include <cusolverDn.h>
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
    // Enable the optimization: replace TRSM with POTRI(+POTRF upper) + TRMM
    // This computes a one-off triangular inverse equivalent and applies it via
    // TRMM.
    constexpr bool kUsePotriTrmm = false;

    if (n <= nb) {
        return matrix_ops::cusolver::potrf<T>(handle, A, n, lda);
    }

    lda = lda == 0 ? n : lda;

    auto cublas_handle = common::CublasHandle();

    // Setup cuSOLVER generic API params and data type metadata once
    cusolverDnParams_t params = nullptr;
    cusolverDnCreateParams(&params);

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
                      "potrf only supports float and double");
    }

    for (auto outer_index = 0; outer_index < n; outer_index += nb) {
        for (auto inner_index = outer_index;
             inner_index < outer_index + nb - 1 && inner_index < n;
             inner_index += b) {
            // panel factorization
            matrix_ops::cusolver::potrf(
                handle, A + inner_index * lda + inner_index, b, lda);
            if (inner_index + b >= n) break;

            // update the elements below the panel
            // ops = (n - inner_index - b) * b * b;

            const int m_total = static_cast<int>(n - inner_index - b);
            const int nb_int = static_cast<int>(b);
            if (kUsePotriTrmm && m_total > 0) {
                // 1) Copy the diagonal block L_kk (b x b) to a temp buffer
                thrust::device_vector<T> diag_tmp(b * b);
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    A + inner_index * lda + inner_index, lda, diag_tmp.data(),
                    b, b, b);

                // 2) POTRI on the copied block to form A_kk^{-1}
                thrust::device_vector<int> info_pi(1);
                if constexpr (std::is_same_v<T, float>) {
                    int potri_lwork = 0;
                    cusolverDnSpotri_bufferSize(
                        handle, CUBLAS_FILL_MODE_LOWER, nb_int,
                        thrust::raw_pointer_cast(diag_tmp.data()), nb_int,
                        &potri_lwork);
                    thrust::device_vector<float> potri_work(potri_lwork);
                    cusolverDnSpotri(
                        handle, CUBLAS_FILL_MODE_LOWER, nb_int,
                        thrust::raw_pointer_cast(diag_tmp.data()), nb_int,
                        thrust::raw_pointer_cast(potri_work.data()),
                        potri_lwork, thrust::raw_pointer_cast(info_pi.data()));
                } else if constexpr (std::is_same_v<T, double>) {
                    int potri_lwork = 0;
                    cusolverDnDpotri_bufferSize(
                        handle, CUBLAS_FILL_MODE_LOWER, nb_int,
                        thrust::raw_pointer_cast(diag_tmp.data()), nb_int,
                        &potri_lwork);
                    thrust::device_vector<double> potri_work(potri_lwork);
                    cusolverDnDpotri(
                        handle, CUBLAS_FILL_MODE_LOWER, nb_int,
                        thrust::raw_pointer_cast(diag_tmp.data()), nb_int,
                        thrust::raw_pointer_cast(potri_work.data()),
                        potri_lwork, thrust::raw_pointer_cast(info_pi.data()));
                }

                // 3) Cholesky on A_kk^{-1} with LOWER fill to get Lhat s.t.
                // Lhat*Lhat^T = A_kk^{-1}
                //    Note: Lhat = L_kk^{-1}
                size_t size_d_pf = 0, size_h_pf = 0;
                cusolverDnXpotrf_bufferSize(
                    handle, params, CUBLAS_FILL_MODE_LOWER, nb_int, data_type,
                    diag_tmp.data().get(), nb_int, compute_type, &size_d_pf,
                    &size_h_pf);
                thrust::device_vector<char> work_d_pf(size_d_pf);
                thrust::host_vector<char> work_h_pf(size_h_pf);
                thrust::device_vector<int> info_pf(1);
                cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, nb_int,
                                 data_type, diag_tmp.data().get(), nb_int,
                                 compute_type, work_d_pf.data().get(),
                                 size_d_pf, work_h_pf.data(), size_h_pf,
                                 info_pf.data().get());

                // 4) TRMM on the tall block: B := B * (Lhat^T) = B * L_kk^{-T}
                //    Use LOWER + op(A)=T on Lhat to get (Lhat)^T.
                auto B_in = A + (inner_index + b) + inner_index * lda;
                thrust::device_vector<T> B_out(static_cast<size_t>(m_total) *
                                               static_cast<size_t>(nb_int));
                if constexpr (std::is_same_v<T, float>) {
                    const float alpha = 1.0f;
                    cublasStrmm(
                        cublas_handle, CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT, m_total, nb_int, &alpha,
                        (const float*)thrust::raw_pointer_cast(diag_tmp.data()),
                        nb_int, thrust::raw_pointer_cast(B_in), lda,
                        thrust::raw_pointer_cast(B_out.data()), m_total);
                } else if constexpr (std::is_same_v<T, double>) {
                    const double alpha = 1.0;
                    cublasDtrmm(cublas_handle, CUBLAS_SIDE_RIGHT,
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, m_total, nb_int, &alpha,
                                (const double*)thrust::raw_pointer_cast(
                                    diag_tmp.data()),
                                nb_int, thrust::raw_pointer_cast(B_in), lda,
                                thrust::raw_pointer_cast(B_out.data()),
                                m_total);
                }
                // Copy B_out back to in-place position
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    B_out.data(), m_total, B_in, lda, m_total, nb_int);
            } else {
                if constexpr (std::is_same_v<T, float>) {
                    float alpha = 1.0f;
                    cublasStrsm(cublas_handle, CUBLAS_SIDE_RIGHT,
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, m_total, nb_int, &alpha,
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
                                CUBLAS_DIAG_NON_UNIT, m_total, nb_int, &alpha,
                                (const double*)thrust::raw_pointer_cast(
                                    A + inner_index * lda + inner_index),
                                lda,
                                thrust::raw_pointer_cast(A + (inner_index + b) +
                                                         inner_index * lda),
                                lda);
                }
            }
            if (inner_index + b >= outer_index + nb - 1) break;

            // update the following panel
            constexpr bool kUseGemmForPanelUpdate = true;
            // When logging is disabled, fuse the two GEMMs into one to reduce
            // launches.
            if constexpr (kUseGemmForPanelUpdate) {
                const auto k =
                    static_cast<size_t>(inner_index + b - outer_index);
                const auto m_total = static_cast<size_t>(n - inner_index - b);
                if (m_total > 0 && k > 0) {
                    constexpr T alpha = static_cast<T>(-1.0);
                    constexpr T beta = static_cast<T>(1.0);
                    // Fused update: Cblk(inner_index+b : n-1, inner_index+b :
                    // inner_index+2b-1) = beta*Cblk + alpha * W * X^T where W =
                    // A(inner_index+b : n-1, outer_index : inner_index+b-1)
                    //       X = A(inner_index+b : inner_index+2b-1, outer_index
                    //       : inner_index+b-1)
                    matrix_ops::gemm(
                        cublas_handle, m_total, b, k, alpha,
                        A + (inner_index + b) + outer_index * lda, lda, false,
                        A + (inner_index + b) + outer_index * lda, lda, true,
                        beta, A + (inner_index + b) + (inner_index + b) * lda,
                        lda);
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
                            cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            b, inner_index + b - outer_index, &alpha,
                            thrust::raw_pointer_cast(A + (inner_index + b) +
                                                     outer_index * lda),
                            lda, &beta,
                            thrust::raw_pointer_cast(A + inner_index + b +
                                                     (inner_index + b) * lda),
                            lda);
                    } else if constexpr (std::is_same_v<T, double>) {
                        double alpha = -1.0;
                        double beta = 1.0;
                        cublasDsyrk(
                            cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            b, inner_index + b - outer_index, &alpha,
                            thrust::raw_pointer_cast(A + (inner_index + b) +
                                                     outer_index * lda),
                            lda, &beta,
                            thrust::raw_pointer_cast(A + inner_index + b +
                                                     (inner_index + b) * lda),
                            lda);
                    }
                }

                if (inner_index + 2 * b >= n) continue;

                matrix_ops::gemm(
                    cublas_handle, n - inner_index - 2 * b, b,
                    inner_index - outer_index + b, (T)-1.0,
                    A + (inner_index + 2 * b) + outer_index * lda, lda, false,
                    A + (inner_index + b) + outer_index * lda, lda, true,
                    (T)1.0, A + (inner_index + 2 * b) + (inner_index + b) * lda,
                    lda);
            }
        }

        if (outer_index + nb >= n) break;

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

        if (outer_index + 2 * nb >= n) continue;

        matrix_ops::gemm(cublas_handle, n - outer_index - 2 * nb, nb,
                         outer_index + nb, (T)-1.0, A + outer_index + 2 * nb,
                         lda, false, A + outer_index + nb, lda, true, (T)1.0,
                         A + outer_index + 2 * nb + (outer_index + nb) * lda,
                         lda);
    }

    if (params) {
        cusolverDnDestroyParams(params);
    }

    return 0;
}
}  // namespace osla
}  // namespace matrix_ops