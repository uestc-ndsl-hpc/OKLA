#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <type_traits>

#include "../src/cusolver_warppers/cusolver_warppers.cuh"
#include "../src/matrix_ops/matrix_ops.cuh"
#include "../src/okla_warppers/okla_warpper.cuh"

template <typename T>
struct DiagonalShiftFunctor {
    T* data;
    size_t n;

    __device__ void operator()(size_t i) const {
        data[i * n + i] += static_cast<T>(n);
    }
};

template <typename T>
struct ZeroUpperTriangleFunctor {
    T* data;
    size_t n;
    size_t lda;

    __device__ void operator()(size_t k) const {
        size_t row = k % n;
        size_t col = k / n;
        if (row < col) {
            data[col * lda + row] = static_cast<T>(0);
        }
    }
};

template <typename T>
void RunPotrfReturnsSuccess() {
    constexpr size_t n = 4096;
    auto d_A = matrix_ops::create_symmetric_random<T>(n);
    // Ensure positive definiteness by shifting the diagonal on device.
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        DiagonalShiftFunctor<T>{thrust::raw_pointer_cast(d_A.data()), n});

    constexpr const char* kTypeName =
        std::is_same_v<T, float> ? "float" : "double";
    EXPECT_EQ(matrix_ops::okla::potrf<T>(d_A.data(), n),
              matrix_ops::okla::OklaStatus::OKLA_STATUS_SUCCESS)
        << "OKLA potrf<" << kTypeName << "> should return OKLA_STATUS_SUCCESS";
}

/**
 * @brief Test for OKLA potrf function
 */
TEST(OklaTest, PotrfReturnsSuccess) {
    RunPotrfReturnsSuccess<float>();
    RunPotrfReturnsSuccess<double>();
}

template <typename T>
void RunPotrfMatchesCusolver() {
    constexpr size_t n = 64;
    auto d_A = matrix_ops::create_symmetric_random<T>(n);
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        DiagonalShiftFunctor<T>{thrust::raw_pointer_cast(d_A.data()), n});

    thrust::device_vector<T> d_A_okla = d_A;
    thrust::device_vector<T> d_A_cusolver = d_A;

    util::Logger::tic("Naive implement with cuSOLVERDx");
    auto okla_status = matrix_ops::okla::potrf<T>(d_A_okla.data(), n);
    util::Logger::toc("Naive implement with cuSOLVERDx", n * n * n);
    ASSERT_EQ(okla_status, matrix_ops::okla::OklaStatus::OKLA_STATUS_SUCCESS);

    auto cuda_status = cudaDeviceSynchronize();
    ASSERT_EQ(cuda_status, cudaSuccess) << "cudaDeviceSynchronize failed";

    common::CusolverDnHandle cusolver_handle;
    util::Logger::tic("cuSOLVER potrf");
    auto cusolver_status =
        matrix_ops::cusolver::potrf<T>(cusolver_handle, d_A_cusolver.data(), n);
    ASSERT_EQ(cusolver_status, 0) << "cusolver::potrf reported failure";
    util::Logger::toc("cuSOLVER potrf", n * n * n);


    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     ZeroUpperTriangleFunctor<T>{
                         thrust::raw_pointer_cast(d_A_okla.data()), n, n});
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     ZeroUpperTriangleFunctor<T>{
                         thrust::raw_pointer_cast(d_A_cusolver.data()), n, n});

    thrust::device_vector<T> diff(n * n);
    thrust::transform(d_A_okla.begin(), d_A_okla.end(), d_A_cusolver.begin(),
                      diff.begin(),
                      [] __device__(T lhs, T rhs) { return lhs - rhs; });

    common::CublasHandle cublas_handle;
    T norm = static_cast<T>(0);
    if constexpr (std::is_same_v<T, float>) {
        cublasSnrm2(cublas_handle, n * n, thrust::raw_pointer_cast(diff.data()),
                    1, &norm);
    } else if constexpr (std::is_same_v<T, double>) {
        cublasDnrm2(cublas_handle, n * n, thrust::raw_pointer_cast(diff.data()),
                    1, &norm);
    }

    T tolerance = static_cast<T>(1e-4);
    if constexpr (std::is_same_v<T, double>) {
        tolerance = static_cast<T>(1e-12);
    }
    EXPECT_LE(norm / static_cast<T>(n), tolerance)
        << "OKLA potrf does not match cuSOLVER potrf within tolerance";
}

TEST(OklaTest, PotrfMatchesCusolver) {
    RunPotrfMatchesCusolver<float>();
    RunPotrfMatchesCusolver<double>();
}

/**
 * @brief Test that OKLA status codes are correctly defined
 */
TEST(OklaTest, StatusCodesAreCorrect) {
    using matrix_ops::okla::OklaStatus;

    EXPECT_EQ(static_cast<int>(OklaStatus::OKLA_STATUS_SUCCESS), 0);
    EXPECT_EQ(static_cast<int>(OklaStatus::OKLA_STATUS_INVALID_VALUE), -1);
    EXPECT_EQ(static_cast<int>(OklaStatus::OKLA_STATUS_EXECUTION_FAILED), -2);
    EXPECT_EQ(static_cast<int>(OklaStatus::OKLA_STATUS_NOT_INITIALIZED), -3);
}
