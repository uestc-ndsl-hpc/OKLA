#include <argh.h>
#include <fcntl.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>

#include "common/log.h"
#include "cusolver_warppers/cusolver_warppers.cuh"
#include "matrix_ops/matrix_ops.cuh"
#include "osla_warppers/osla_warppers.cuh"
#include "osla_warppers/trsm_wapppers.cuh"

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
int benchmark(argh::parser& cmdl, size_t n, size_t m, size_t nrhs, size_t nb,
              size_t b) {
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
    auto cublas_handle = common::CublasHandle();

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

    if (cmdl[{"--test-cusolver-trsm"}] || cmdl[{"--test-osla-trsm"}]) {
        auto A_trsm = matrix_ops::create_uniform_random<T>(m, m);
        thrust::for_each(thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(m * m),
                         [A_ptr = A_trsm.data(), m] __device__(size_t k) {
                             size_t row = k % m;
                             size_t col = k / m;
                             if (row < col) {
                                 A_ptr[col * m + row] = static_cast<T>(0);
                             }
                         });
        thrust::for_each(thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(m),
                         [A_ptr = A_trsm.data(), m] __device__(size_t i) {
                             A_ptr[i * m + i] += static_cast<T>(m);
                         });

        auto B = matrix_ops::create_uniform_random<T>(m, nrhs);
        thrust::device_vector<T> B0 = B;

        float ops = static_cast<float>(m) * static_cast<float>(m) *
                    static_cast<float>(nrhs);

        if (cmdl[{"--test-cusolver-trsm"}]) {
            B = B0;
            util::Logger::tic("Cusolver TRSM");
            auto status = matrix_ops::cusolver::trsm(
                cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, static_cast<T>(1.0),
                A_trsm.data(), B.data(), m, m);
            util::Logger::toc("Cusolver TRSM", ops);

            if (status != 0) {
                std::cerr << "TRSM failed with status: " << status << std::endl;
                return -1;
            }
        }

        if (cmdl[{"--test-osla-trsm"}]) {
            B = B0;
            util::Logger::tic("OSLA TRSM");
            auto status = matrix_ops::osla::trsm(
                cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, static_cast<T>(1.0),
                A_trsm.data(), B.data(), m, m, nb, b);
            util::Logger::toc("OSLA TRSM", ops);

            if (status != 0) {
                std::cerr << "OSLA TRSM failed with status: " << status
                          << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    // cli args parse
    argh::parser cmdl(argv);
    auto n = (size_t)8192;
    auto m = n;
    auto nrhs = n;
    auto nb = (size_t)8192;
    auto b = (size_t)64;
    cmdl({"--size"}, n) >> n;
    cmdl({"--m"}, m) >> m;
    cmdl({"--nrhs"}, nrhs) >> nrhs;
    cmdl({"--nb"}, nb) >> nb;
    cmdl({"--b"}, b) >> b;
    auto verbose = cmdl[{"-v", "--verbose"}];
    util::Logger::init(verbose);
    util::Logger::init_timer(verbose);
    util::Logger::print_environment_info();

    if (cmdl[{"--float"}]) {
        return benchmark<float>(cmdl, n, m, nrhs, nb, b);
    } else if (cmdl[{"--double"}]) {
        return benchmark<double>(cmdl, n, m, nrhs, nb, b);
    } else {
        std::cerr << "Please specify --float or --double for the data type."
                  << std::endl;
        return -1;
    }

    return 0;
}
