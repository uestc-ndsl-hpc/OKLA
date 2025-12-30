#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "../src/common/handle_warppers.h"
#include "../src/cusolver_warppers/trsm_warpper.cuh"
#include "../src/matrix_ops/matrix_ops.cuh"
#include "../src/okla_warppers/trsm_warppers.cuh"
#include "../src/osla_warppers/trsm_warppers.cuh"

template <typename T>
struct zero_upper_triangle_functor {
    T* A_ptr_;
    size_t n_;
    size_t lda_;

    zero_upper_triangle_functor(thrust::device_ptr<T> A, size_t n, size_t lda)
        : A_ptr_(thrust::raw_pointer_cast(A)), n_(n), lda_(lda) {}

    __device__ void operator()(const size_t& k) const {
        size_t row = k % n_;
        size_t col = k / n_;
        size_t physical_index = col * lda_ + row;

        if (row < col) {
            A_ptr_[physical_index] = static_cast<T>(0);
        }
    }
};

template <typename T>
struct boost_diag_functor {
    T* A_ptr_;
    size_t n_;
    size_t lda_;
    T shift_;

    boost_diag_functor(thrust::device_ptr<T> A, size_t n, size_t lda, T shift)
        : A_ptr_(thrust::raw_pointer_cast(A)),
          n_(n),
          lda_(lda),
          shift_(shift) {}

    __device__ void operator()(const size_t& k) const {
        A_ptr_[k * lda_ + k] += shift_;
    }
};

template <typename T>
T max_abs_diff(const thrust::device_vector<T>& a,
               const thrust::device_vector<T>& b) {
    thrust::host_vector<T> h_a = a;
    thrust::host_vector<T> h_b = b;
    T max_val = static_cast<T>(0);
    for (size_t i = 0; i < h_a.size(); ++i) {
        T diff = std::abs(h_a[i] - h_b[i]);
        max_val = std::max(max_val, diff);
    }
    return max_val;
}

template <typename T>
T max_abs(const thrust::device_vector<T>& a) {
    thrust::host_vector<T> h_a = a;
    T max_val = static_cast<T>(0);
    for (size_t i = 0; i < h_a.size(); ++i) {
        T val = std::abs(h_a[i]);
        max_val = std::max(max_val, val);
    }
    return max_val;
}

template <typename T>
void RunTrsmBlockedTest() {
    constexpr size_t n = 27;
    constexpr size_t nb = 14;
    // constexpr size_t b = 8;

    T tolerance = static_cast<T>(1e-4);
    if constexpr (std::is_same_v<T, double>) {
        tolerance = static_cast<T>(1e-10);
    }

    auto d_A = matrix_ops::create_uniform_random<T>(n, n);
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     zero_upper_triangle_functor<T>(d_A.data(), n, n));
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        boost_diag_functor<T>(d_A.data(), n, n, static_cast<T>(n)));

    auto d_X = matrix_ops::create_uniform_random<T>(n, n);

    common::CublasHandle handle;
    thrust::device_vector<T> d_B(n * n);
    matrix_ops::gemm(handle, n, n, n, static_cast<T>(1.0), d_A.data(), n,
                     d_X.data(), n, static_cast<T>(0.0), d_B.data(), n);

    thrust::device_vector<T> d_B0 = d_B;

    matrix_ops::osla::tensorblas::trsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n,
                           static_cast<T>(1.0), d_A.data(), d_B.data(), n, n,
                           nb);

    const T max_solution_error = max_abs_diff(d_B, d_X);
    ASSERT_LE(max_solution_error, tolerance)
        << "Blocked TRSM result does not match the expected solution.";

    thrust::device_vector<T> d_residual = d_B0;
    matrix_ops::gemm(handle, n, n, n, static_cast<T>(1.0), d_A.data(), n,
                     d_B.data(), n, static_cast<T>(-1.0), d_residual.data(), n);
    const T max_residual = max_abs(d_residual);
    ASSERT_LE(max_residual, tolerance)
        << "A * X did not reconstruct the original B within tolerance.";
}

TEST(TrsmBlockedTest, SolvesLowerTriangularFloat) {
    RunTrsmBlockedTest<float>();
}

TEST(TrsmBlockedTest, SolvesLowerTriangularDouble) {
    RunTrsmBlockedTest<double>();
}

template <typename T>
void RunTrsmOklaTest() {
    constexpr size_t n = 128;
    constexpr T alpha = static_cast<T>(1.0);
    constexpr T tolerance = static_cast<T>(1e-3);

    auto d_A = matrix_ops::create_uniform_random<T>(n, n);
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     zero_upper_triangle_functor<T>(d_A.data(), n, n));
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        boost_diag_functor<T>(d_A.data(), n, n, static_cast<T>(n)));

    auto d_X = matrix_ops::create_uniform_random<T>(n, n);

    common::CublasHandle handle;
    thrust::device_vector<T> d_B(n * n);
    matrix_ops::gemm(handle, n, n, n, static_cast<T>(1.0), d_A.data(), n,
                     d_X.data(), n, static_cast<T>(0.0), d_B.data(), n);

    thrust::device_vector<T> d_B0 = d_B;

    int status = matrix_ops::okla::trsm(
        handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, n, alpha, d_A.data(), d_B.data(), n, n);
    ASSERT_EQ(status, 0) << "OKLA TRSM dispatch failed or was not matched.";

    const T max_solution_error = max_abs_diff(d_B, d_X);
    ASSERT_LE(max_solution_error, tolerance)
        << "OKLA TRSM result does not match the expected solution.";

    thrust::device_vector<T> d_residual = d_B0;
    matrix_ops::gemm(handle, n, n, n, static_cast<T>(1.0), d_A.data(), n,
                     d_B.data(), n, static_cast<T>(-1.0), d_residual.data(), n);
    const T max_residual = max_abs(d_residual);
    ASSERT_LE(max_residual, tolerance)
        << "A * X did not reconstruct the original B within tolerance.";
}

TEST(TrsmOklaTest, SolvesLowerTriangularFloat) {
    RunTrsmOklaTest<float>();
}

TEST(TrsmOklaTest, MatchesCublasFloatAlpha1) {
    constexpr size_t m = 128;
    constexpr float alpha = 1.0f;
    constexpr float tolerance = 1e-3f;

    auto d_A = matrix_ops::create_uniform_random<float>(m, m);
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(m * m),
                     zero_upper_triangle_functor<float>(d_A.data(), m, m));
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(m),
        boost_diag_functor<float>(d_A.data(), m, m, static_cast<float>(m)));

    constexpr size_t n_cases[] = {1,   7,   31,  32,  33,  63,  64,
                                  65,  127, 128, 129, 255, 256, 257,
                                  511, 512, 513, 1024, 2048, 4096};

    common::CublasHandle handle;
    for (size_t n : n_cases) {
        auto d_B0 = matrix_ops::create_uniform_random<float>(m, n);

        auto d_B_okla = d_B0;
        int okla_status = matrix_ops::okla::trsm(
            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, m, n, alpha, d_A.data(), d_B_okla.data(), m,
            m);
        ASSERT_EQ(okla_status, 0) << "okla trsm dispatch failed for n=" << n;

        auto d_B_cublas = d_B0;
        int cublas_status = matrix_ops::cusolver::trsm(
            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, m, n, alpha, d_A.data(), d_B_cublas.data(), m,
            m);
        ASSERT_EQ(cublas_status, 0) << "cublas trsm failed for n=" << n;

        float max_solution_error = max_abs_diff(d_B_okla, d_B_cublas);
        ASSERT_LE(max_solution_error, tolerance)
            << "okla trsm mismatch vs cublas for n=" << n;
    }
}
