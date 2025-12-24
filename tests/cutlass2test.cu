#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/device_memory.h>
#include <thrust/device_vector.h>
#include "fmt/base.h"

#define CUTLASS_CHECK(status)                                                 \
    {                                                                         \
        cutlass::Status error = status;                                       \
        if (error != cutlass::Status::kSuccess) {                             \
            std::cerr << "Got cutlass error: "                                \
                      << cutlassGetStatusString(error) << " at: " << __LINE__ \
                      << std::endl;                                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

/**
you can configure ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp,
SwizzleThreadBlock and Numstages as well.
*/
using Gemm = cutlass::gemm::device::Gemm<
    double, cutlass::layout::ColumnMajor, double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor, double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;

constexpr int iterations = 10;

int main() {
    cutlass::gemm::GemmCoord problem_size(16384, 16384, 16384);
    thrust::device_vector<double> dA(problem_size.m() * problem_size.k());
    thrust::device_vector<double> dB(problem_size.k() * problem_size.n());
    thrust::device_vector<double> dC(problem_size.m() * problem_size.n());

    thrust::fill(dA.begin(), dA.end(), 1.0);
    thrust::fill(dB.begin(), dB.end(), 1.0);
    thrust::fill(dC.begin(), dC.end(), 0.0);

    cutlass::layout::ColumnMajor layoutA(
        problem_size.m());  // A: MxK, ColumnMajor -> ld = M (rows)
    cutlass::layout::ColumnMajor layoutB(
        problem_size.k());  // B: KxN, ColumnMajor -> ld = K (rows)
    cutlass::layout::ColumnMajor layoutC(
        problem_size.m());  // C: MxN, ColumnMajor -> ld = M (rows)

    cutlass::TensorRef<double, cutlass::layout::ColumnMajor> A(
        thrust::raw_pointer_cast(dA.data()), layoutA);
    cutlass::TensorRef<double, cutlass::layout::ColumnMajor> B(
        thrust::raw_pointer_cast(dB.data()), layoutB);
    cutlass::TensorRef<double, cutlass::layout::ColumnMajor> C(
        thrust::raw_pointer_cast(dC.data()), layoutC);
    cutlass::TensorRef<double, cutlass::layout::ColumnMajor> D(
        thrust::raw_pointer_cast(dC.data()), layoutC);

    typename Gemm::Arguments arguments(problem_size, A, B, C, D, {1.0, 0.0}, 1);

    auto workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(
        workspace_size);  // uint8 for one byte

    Gemm gemm_op;

    auto status = gemm_op.can_implement(arguments);

    CUTLASS_CHECK(status);
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (auto i = 0; i < iterations; i++) {
        status = gemm_op.run();
        CUTLASS_CHECK(status);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_ms = double(total_ms) / double(iterations);
    double avg_s  = avg_ms / 1000.0;
    double flops = 2.0 * double(problem_size.m()) * double(problem_size.n()) * double(problem_size.k());
    double tflops = flops / 1.0e12 / avg_s;

    fmt::println("Average time: {:.3f} ms, {:.3f} s, {:.3f} TFLOPS", avg_ms, avg_s, tflops);

    return 0;
}