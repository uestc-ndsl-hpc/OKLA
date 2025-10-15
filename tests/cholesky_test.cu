#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../src/common/handle_warppers.h"
#include "../src/cusolver_warppers/cusolver_warppers.cuh"
#include "../src/matrix_ops/matrix_ops.cuh"
#include "../src/osla_warppers/osla_warppers.cuh"

/**
 * @brief functor to extract L
 */
template <typename T>
struct extract_L_functor_2d {
    T* A_ptr_;
    size_t m_;
    size_t lda_;

    extract_L_functor_2d(thrust::device_ptr<T> A, size_t m, size_t lda)
        : A_ptr_(thrust::raw_pointer_cast(A)), m_(m), lda_(lda) {}

    __device__ void operator()(const size_t& k) const {
        size_t row = k % m_;
        size_t col = k / m_;

        size_t physical_index = col * lda_ + row;

        if (row < col) {
            A_ptr_[physical_index] = 0.0;
        }
    }
};

template <typename T>
void RunCholeskyTest() {
    constexpr size_t n = 4096;
    T kTolerance = static_cast<T>(1e-4);
    if constexpr (std::is_same_v<T, double>) {
        kTolerance = static_cast<T>(1e-13);
    }
    // create positive-definite matrix A
    auto d_A = matrix_ops::create_symmetric_random<T>(n);
    // A = A + n * I
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n),
                     [A_ptr = d_A.data(), n] __device__(size_t i) {
                         A_ptr[i * n + i] += static_cast<T>(n);
                     });

    // keep a host copy of A so we can re-create the device matrix after
    // in-place Cholesky modifies the input
    thrust::host_vector<T> h_A = d_A;

    common::CusolverDnHandle handle;
    common::CublasHandle cublas_handle;
    const auto status = matrix_ops::cusolver::cholesky(handle, d_A.data(), n);
    ASSERT_EQ(status, 0) << "cusolverDnXpotrf reported failure.";
    // verify that norm(A - L * L^T) / n vs tol
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     extract_L_functor_2d<T>(d_A.data(), n, n));
    thrust::device_vector<T> ori_A = h_A;
    matrix_ops::gemm(cublas_handle, n, n, n, (T)-1.0, d_A.data(), n, false,
                     d_A.data(), n, true, (T)1.0, ori_A.data(), n);
    if constexpr (std::is_same_v<T, float>) {
        float norm = 0;
        cublasSnrm2(cublas_handle, n * n,
                    thrust::raw_pointer_cast(ori_A.data()), 1, &norm);
        ASSERT_LE(norm / n, kTolerance)
            << "Cusolver Cholesky factorization result is incorrect.";
    } else if constexpr (std::is_same_v<T, double>) {
        double norm = 0;
        cublasDnrm2(cublas_handle, n * n,
                    thrust::raw_pointer_cast(ori_A.data()), 1, &norm);
        ASSERT_LE(norm / n, kTolerance)
            << "Cusolver Cholesky factorization result is incorrect.";
    }

    // also test OSLA implementation which should call potrf under the hood
    // (uses same API surface)
    // recreate device copy because previous call modified d_A in-place
    d_A = h_A;
    const auto osla_status =
        matrix_ops::osla::cholesky(handle, d_A.data(), n, n, n / 2, n / 4);
    ASSERT_EQ(osla_status, 0) << "osla cholesky reported failure.";
    // verify that norm(A - L * L^T) / n vs tol
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     extract_L_functor_2d<T>(d_A.data(), n, n));
    ori_A = h_A;
    matrix_ops::gemm(cublas_handle, n, n, n, (T)-1.0, d_A.data(), n, false,
                     d_A.data(), n, true, (T)1.0, ori_A.data(), n);
    if constexpr (std::is_same_v<T, float>) {
        float norm = 0;
        cublasSnrm2(cublas_handle, n * n,
                    thrust::raw_pointer_cast(ori_A.data()), 1, &norm);
        ASSERT_LE(norm / n, kTolerance)
            << "OSLA Cholesky factorization result is incorrect.";
    } else if constexpr (std::is_same_v<T, double>) {
        double norm = 0;
        cublasDnrm2(cublas_handle, n * n,
                    thrust::raw_pointer_cast(ori_A.data()), 1, &norm);
        ASSERT_LE(norm / n, kTolerance)
            << "OSLA Cholesky factorization result is incorrect.";
    }
}

TEST(CholeskyTest, FactorizesPositiveDefiniteFloat) {
    RunCholeskyTest<float>();
}

TEST(CholeskyTest, FactorizesPositiveDefiniteDouble) {
    RunCholeskyTest<double>();
}
