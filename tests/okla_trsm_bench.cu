#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../src/okla_warppers/trsm_warppers.cuh"

namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::cerr << "[cuda] " << what << ": " << cudaGetErrorString(status)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_dispatch(int status, const char* what) {
    if (status == -1) {
        std::cerr << "[dispatch] miss: " << what << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (status != cudaSuccess) {
        std::cerr << "[cuda] " << what << ": "
                  << cudaGetErrorString(static_cast<cudaError_t>(status))
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int size = 128;
    const float alpha = 1.0f;

    int warmup_iters = 20;
    int iters = 200;
    if (argc > 1) {
        const int v = std::atoi(argv[1]);
        if (v == 128 || v == 256) {
            size = v;
            if (argc > 2) {
                iters = std::max(1, std::atoi(argv[2]));
            }
            if (argc > 3) {
                warmup_iters = std::max(0, std::atoi(argv[3]));
            }
        } else {
            iters = std::max(1, v);
            if (argc > 2) {
                warmup_iters = std::max(0, std::atoi(argv[2]));
            }
        }
    }
    const int m = size;
    const int n = size;
    const int lda = size;
    const int ldb = size;

    std::vector<float> h_A(static_cast<size_t>(m) * m, 0.0f);
    std::vector<float> h_B(static_cast<size_t>(m) * n, 1.0f);
    for (int col = 0; col < m; ++col) {
        for (int row = col; row < m; ++row) {
            h_A[col * lda + row] =
                (row == col) ? 2.0f : 0.01f * static_cast<float>(row - col);
        }
    }

    thrust::device_vector<float> d_A(h_A.begin(), h_A.end());
    thrust::device_vector<float> d_B(h_B.begin(), h_B.end());

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "stream create");

    auto dispatch = [&]() {
        return matrix_ops::okla::trsm_dispatch_okla<float>(
            stream, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, m, n, alpha,
            thrust::raw_pointer_cast(d_A.data()), lda,
            thrust::raw_pointer_cast(d_B.data()), ldb);
    };

    for (int i = 0; i < warmup_iters; ++i) {
        check_dispatch(dispatch(), "warmup launch");
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "event create start");
    check_cuda(cudaEventCreate(&stop), "event create stop");

    check_cuda(cudaEventRecord(start, stream), "event record start");
    for (int i = 0; i < iters; ++i) {
        check_dispatch(dispatch(), "timed launch");
    }
    check_cuda(cudaEventRecord(stop, stream), "event record stop");
    check_cuda(cudaEventSynchronize(stop), "event sync stop");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "event elapsed");
    const float avg_ms = elapsed_ms / static_cast<float>(iters);
    const double flops =
        static_cast<double>(m) * static_cast<double>(m) * static_cast<double>(n);
    const double tflops = flops / (static_cast<double>(avg_ms) * 1.0e9);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "okla trsm kernel " << m << "x" << n
              << " avg time: " << avg_ms << " ms (" << iters
              << " iters), tflops: " << tflops << std::endl;

    check_cuda(cudaEventDestroy(start), "event destroy start");
    check_cuda(cudaEventDestroy(stop), "event destroy stop");
    check_cuda(cudaStreamDestroy(stream), "stream destroy");

    return 0;
}
