#include <argh.h>
#include <fcntl.h>

#include <cstddef>

#include "common/log.h"
#include "cusolver_warppers/cusolver_warppers.cuh"
#include "matrix_ops/matrix_ops.cuh"
#include "osla_warppers/osla_warppers.cuh"

template <typename T>
void warm_up() {
    if (util::Logger::is_verbose()) {
        util::Logger::println("[info] Performing GEMM warm-up");
    }
    const int n_warmup = 16384;
    thrust::device_vector<T> d_A(n_warmup * n_warmup, 1.0);
    thrust::device_vector<T> d_B(n_warmup * n_warmup, 1.0);
    thrust::device_vector<T> d_C(n_warmup * n_warmup, 0.0);
    T alpha = 1.0;
    T beta = 0.0;
    common::CublasHandle handle;
    for (int i = 0; i < 10; i++) {
        if constexpr (std::is_same_v<T, float>) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_warmup, n_warmup,
                        n_warmup, (const float*)&alpha,
                        (const float*)thrust::raw_pointer_cast(d_A.data()),
                        n_warmup,
                        (const float*)thrust::raw_pointer_cast(d_B.data()),
                        n_warmup, (const float*)&beta,
                        (float*)thrust::raw_pointer_cast(d_C.data()), n_warmup);
        } else {
            cublasDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n_warmup, n_warmup, n_warmup,
                (const double*)&alpha,
                (const double*)thrust::raw_pointer_cast(d_A.data()), n_warmup,
                (const double*)thrust::raw_pointer_cast(d_B.data()), n_warmup,
                (const double*)&beta,
                (double*)thrust::raw_pointer_cast(d_C.data()), n_warmup);
        }
        cudaDeviceSynchronize();
    }
    util::Logger::println("[info] Warm-up finished");
}

template <typename T>
int benchmark(argh::parser& cmdl, size_t n) {
    if (!cmdl[{"--nowarmup"}]) {
        warm_up<T>();
    }

    // create positive-definite matrix A
    auto A = matrix_ops::create_symmetric_random<T>(n);
    // A = A + n * I
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n),
                     [A_ptr = A.data(), n] __device__(size_t i) {
                         A_ptr[i * n + i] += static_cast<T>(n);
                     });

    thrust::host_vector<T> h_A = A;

    auto cusolver_handle = common::CusolverDnHandle();

    if (cmdl[{"--test-cusolver"}]) {
        A = h_A;  // reset A
        util::Logger::tic("Cusolver Cholesky Factorization");
        auto status =
            matrix_ops::cusolver::cholesky(cusolver_handle, A.data(), n);
        util::Logger::toc("Cusolver Cholesky Factorization",
                          (1.0 / 3.0) * n * n * n);

        if (status != 0) {
            std::cerr << "Cholesky factorization failed with status: " << status
                      << std::endl;
            return -1;
        }
    }

    if (cmdl[{"--test-osla"}]) {
        A = h_A;  // reset A
        util::Logger::tic("OSLA Cholesky Factorization");
        auto status = matrix_ops::osla::cholesky(cusolver_handle, A.data(), n,
                                                 n, 8192, 1024);
        util::Logger::toc("OSLA Cholesky Factorization",
                          (1.0 / 3.0) * n * n * n);

        if (status != 0) {
            std::cerr << "Cholesky factorization failed with status: " << status
                      << std::endl;
            return -1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    // cli args parse
    argh::parser cmdl(argv);
    auto n = (size_t)8192;
    cmdl({"-n", "--size"}, 4) >> n;
    auto verbose = cmdl[{"-v", "--verbose"}];
    util::Logger::init(verbose);
    util::Logger::print_environment_info();

    if (cmdl[{"--float"}]) {
        return benchmark<float>(cmdl, n);
    } else if (cmdl[{"--double"}]) {
        return benchmark<double>(cmdl, n);
    } else {
        std::cerr << "Please specify --float or --double for the data type."
                  << std::endl;
        return -1;
    }

    return 0;
}