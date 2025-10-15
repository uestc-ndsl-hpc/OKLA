#pragma once

#include <cublas_v2.h>
#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "../common/log.h"
#include "../common/handle_warppers.h"

// fmt::formatter for cublasStatus_t
template <>
struct fmt::formatter<cublasStatus_t> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const cublasStatus_t& status, FormatContext& ctx) const {
        // 调用我们放在 util 命名空间中的辅助函数
        return fmt::format_to(ctx.out(), "{}",
                              util::cublasGetErrorString(status));
    }
};

namespace matrix_ops {
template <typename T>
thrust::device_vector<T> create_symmetric_random(size_t n,
                                                 bool fixed_seed = false);

template <typename T>
thrust::device_vector<T> create_uniform_random(size_t n);

template <typename T>
thrust::device_vector<T> create_uniform_random(size_t m, size_t n);

template <typename T>
thrust::device_vector<T> create_normal_random(size_t n, T mean = 0.0,
                                              T stddev = 1.0);

template <typename T>
thrust::device_vector<T> create_normal_random(size_t m, size_t n, T mean = 0.0,
                                              T stddev = 1.0);

/**
 * @brief General matrix multiplication.
 *
 * @tparam T
 * @param handle A handle to the cuBLAS library context.
 * @param m The number of rows of matrix A.
 * @param n The number of columns of matrix B.
 * @param k The number of columns of matrix A.
 * @param alpha The scalar alpha.
 * @param A The m x k matrix A.
 * @param lda The leading dimension of matrix A.
 * @param B The k x n matrix B.
 * @param ldb The leading dimension of matrix B.
 * @param beta The scalar beta.
 * @param C The m x n matrix C.
 * @param ldc The leading dimension of matrix C.
 */
template <typename T>
void gemm(const common::CublasHandle& handle, size_t m, size_t n, size_t k,
          T alpha, thrust::device_ptr<T> A, size_t lda, thrust::device_ptr<T> B,
          size_t ldb, T beta, thrust::device_ptr<T> C, size_t ldc);

/**
 * @brief General matrix multiplication.
 *
 * @tparam T
 * @param handle A handle to the cuBLAS library context.
 * @param m The number of rows of matrix A.
 * @param n The number of columns of matrix B.
 * @param k The number of columns of matrix A.
 * @param alpha The scalar alpha.
 * @param A The m x k matrix A.
 * @param lda The leading dimension of matrix A.
 * @param transA Whether to transpose matrix A.
 * @param B The k x n matrix B.
 * @param ldb The leading dimension of matrix B.
 * @param transB Whether to transpose matrix B.
 * @param beta The scalar beta.
 * @param C The m x n matrix C.
 * @param ldc The leading dimension of matrix C.
 */
template <typename T>
void gemm(const common::CublasHandle& handle, size_t m, size_t n, size_t k,
          T alpha, thrust::device_ptr<T> A, size_t lda, bool transA,
          thrust::device_ptr<T> B, size_t ldb, bool transB, T beta,
          thrust::device_ptr<T> C, size_t ldc);

/**
 * @brief copy matrix from src to dst
 *
 * @tparam srcPtr source matrix ptr type
 * @tparam dstPtr destination matrix ptr type
 * @param src source matrix ptr
 * @param src_ld source matrix leading dimension
 * @param dst destination matrix ptr
 * @param dst_ld destination matrix leading dimension
 * @param m number of rows
 * @param n number of columns
 */
template <typename srcPtr, typename dstPtr, typename T>
void matrix_copy(srcPtr src, size_t src_ld, dstPtr dst, size_t dst_ld, size_t m,
                 size_t n);

/**
 * @brief print matrix for row = m, col = n, and the input is host ptr (column
 * major and lda provided)
 *
 * @tparam T type of the matrix elements
 * @param data host ptr
 * @param m number of rows
 * @param n number of columns
 * @param lda leading dimension of the matrix
 * @param title title of the matrix
 */
template <typename T>
void print(T* data, size_t m, size_t n, size_t lda, const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is host ptr
 *
 * @tparam T
 * @param data host ptr
 * @param m number of rows
 * @param n number of columns
 * @param title title of the matrix
 */
template <typename T>
void print(T* data, size_t m, size_t n, const std::string& title);

/**
 * @brief print matrix for row = col = n (column major)
 *
 * @tparam T
 * @param d_vec device vector
 * @param n number of rows
 * @param title title of the matrix
 */
template <typename T>
void print(thrust::device_vector<T>& d_vec, size_t n, const std::string& title);

/**
 * @brief print matrix for row = m, col = n (column major)
 *
 * @tparam T
 * @param d_vec device vector
 * @param m number of rows
 * @param n number of columns
 * @param title title of the matrix
 */
template <typename T>
void print(thrust::device_vector<T>& d_vec, size_t m, size_t n,
           const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is device ptr (column
 * major)
 *
 * @tparam T
 * @param data device ptr
 * @param m number of rows
 * @param n number of columns
 * @param title title of the matrix
 */
template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n,
           const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is device ptr (column
 * major and lda provided)
 *
 * @tparam T
 * @param data
 * @param m
 * @param n
 * @param lda
 * @param title
 */
template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n, size_t lda,
           const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is device ptr (column
 * major and lda provided)
 *
 * @tparam T
 * @param h_vec
 * @param m
 * @param n
 * @param lda
 * @param title
 */
template <typename T>
void print(thrust::device_vector<T> h_vec, size_t m, size_t n, size_t lda,
           const std::string& title);
}  // namespace matrix_ops