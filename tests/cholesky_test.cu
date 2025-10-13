#include <cmath>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../src/common/handle_warppers.h"
#include "../src/cusolver_warppers/cusolver_warppers.cuh"

namespace {

template <typename T>
void RunCholeskyTest() {
    constexpr size_t n = 2;
    constexpr T kTolerance = static_cast<T>(1e-5);
    const T sqrt_two = static_cast<T>(std::sqrt(2.0));

    // clang-format off
    thrust::host_vector<T> h_A = {
        static_cast<T>(4.0), static_cast<T>(2.0),
        static_cast<T>(2.0), static_cast<T>(3.0)
    };
    // clang-format on

    thrust::device_vector<T> d_A = h_A;

    common::CusolverDnHandle handle;
    const auto status = matrix_ops::cusolver::cholesky(handle, d_A.data(), n);

    ASSERT_EQ(status, 0) << "cusolverDnXpotrf reported failure.";

    const thrust::host_vector<T> h_result = d_A;
    EXPECT_NEAR(h_result[0], static_cast<T>(2.0), kTolerance);
    EXPECT_NEAR(h_result[1], static_cast<T>(1.0), kTolerance);
    // Upper triangle is unspecified when requesting the lower factor.
    EXPECT_NEAR(h_result[3], sqrt_two, kTolerance);
}

}  // namespace

TEST(CholeskyTest, FactorizesPositiveDefiniteFloat) { RunCholeskyTest<float>(); }

TEST(CholeskyTest, FactorizesPositiveDefiniteDouble) {
    RunCholeskyTest<double>();
}
