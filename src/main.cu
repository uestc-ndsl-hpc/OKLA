#include <argh.h>
#include <fcntl.h>

#include <cstddef>

#include "common/log.h"
#include "cusolver_warppers/cusolver_warppers.cuh"
#include "matrix_ops/matrix_ops.cuh"

using demo_type = float;

int main(int argc, char** argv) {
    // cli args parse
    argh::parser cmdl(argv);
    auto n = (size_t)8192;
    cmdl({"-n", "--size"}, 4) >> n;
    auto verbose = cmdl[{"-v", "--verbose"}];
    util::Logger::init(verbose);

    // create positive-definite matrix A
    auto A = matrix_ops::create_symmetric_random<demo_type>(n);
    // A = A + n * I
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n),
                     [A_ptr = A.data(), n] __device__(size_t i) {
                         A_ptr[i * n + i] += static_cast<demo_type>(n);
                     });

    auto cusolver_handle = common::CusolverDnHandle();

    util::Logger::tic("Cusolver Cholesky Factorization");
    auto status = matrix_ops::cusolver::cholesky(cusolver_handle, A.data(), n);
    util::Logger::toc("Cusolver Cholesky Factorization");

    if (status != 0) {
        std::cerr << "Cholesky factorization failed with status: " << status
                  << std::endl;
        return -1;
    }

    return 0;
}