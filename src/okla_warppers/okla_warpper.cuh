#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cstddef>
#include <cusolverdx.hpp>

namespace matrix_ops {
namespace okla {
// enum for status code of okla
enum class OklaStatus {
    OKLA_STATUS_SUCCESS = 0,
    OKLA_STATUS_INVALID_VALUE = -1,
    OKLA_STATUS_EXECUTION_FAILED = -2,
    OKLA_STATUS_NOT_INITIALIZED = -3
};

namespace common {

constexpr int Arch = 890;

// Convenience wrapper for cusolverdx::copy_2d functions to use in the examples
template <class Operation, unsigned BPB = 1>
struct io {
    using data_type = typename Operation::a_data_type;

    static constexpr unsigned int m = Operation::m_size;
    static constexpr unsigned int n = Operation::n_size;
    static constexpr unsigned int nrhs = Operation::k_size;
    static constexpr unsigned int nthreads = Operation::max_threads_per_block;

    // load BPB batches of A from global memory to shared memory
    static inline __device__ void load_a(const data_type* A, const int lda,
                                         data_type* As, const int ldas) {
        cusolverdx::copy_2d<Operation, m, n,
                            cusolverdx::arrangement_of_v_a<Operation>, BPB>(
            A, lda, As, ldas);
        __syncthreads();
    }

    // store BPB batches of A from shared memory to global memory
    static inline __device__ void store_a(const data_type* As, const int ldas,
                                          data_type* A, const int lda) {
        __syncthreads();
        cusolverdx::copy_2d<Operation, m, n,
                            cusolverdx::arrangement_of_v_a<Operation>, BPB>(
            As, ldas, A, lda);
    }

    // load BPB batches of B from global memory to shared memory
    // Note that the wrapper function cannot be used for unmlq/unmqr function
    // with right size multiplication
    static inline __device__ void load_b(const data_type* B, const int ldb,
                                         data_type* Bs, const int ldbs) {
        cusolverdx::copy_2d<Operation, (m > n ? m : n), nrhs,
                            cusolverdx::arrangement_of_v_b<Operation>, BPB>(
            B, ldb, Bs, ldbs);
        __syncthreads();
    }

    // store BPB batches of B from shared memory to global memory
    // Note that the wrapper function cannot be used for unmlq/unmqr function
    // with right size multiplication
    static inline __device__ void store_b(const data_type* Bs, const int ldbs,
                                          data_type* B, const int ldb) {
        __syncthreads();
        cusolverdx::copy_2d<Operation, (m > n ? m : n), nrhs,
                            cusolverdx::arrangement_of_v_b<Operation>, BPB>(
            Bs, ldbs, B, ldb);
    }
};

}  // namespace common

template <class Solver, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void potrf_kernel(
    DataType* A, const unsigned int lda_smem,
    typename Solver::status_type* info) {
    extern __shared__ unsigned char shared_mem[];
    DataType* As = reinterpret_cast<DataType*>(shared_mem);

    constexpr auto lda_gmem = Solver::m_size;

    // Load data from global memory to shared memory
    common::io<Solver>::load_a(A, lda_gmem, As, lda_smem);

    Solver().execute(As, lda_smem, info);

    // Store results back to global memory
    common::io<Solver>::store_a(As, lda_smem, A, lda_gmem);
}

template <typename T>
OklaStatus potrf(thrust::device_ptr<T> A, size_t n, size_t lda = 0) {
    if (lda == 0) {
        lda = n;
    }

    using namespace cusolverdx;
    // Build a Solver description that matches the template data type T so that
    // the cusolverdx block_execution uses the same data type (float/double)
    // as the matrix provided by the caller. Previously Precision<double>()
    // was hard-coded which caused DataType mismatches when T=float.
    using Solver =
        decltype(Size<64, 64>() + Precision<T>() + Type<type::real>() +
                 Function<function::potrf>() + FillMode<lower>() + Block() +
                 SM<common::Arch>() + BlockDim<64>());
    auto lda_smem = (unsigned int)66;
    const unsigned int sm_size = Solver::get_shared_memory_size(lda_smem);
    thrust::device_vector<typename Solver::status_type> d_info(1);

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    potrf_kernel<Solver><<<1, Solver::block_dim, sm_size, stream>>>(
        thrust::raw_pointer_cast(A), lda_smem,
        thrust::raw_pointer_cast(d_info.data()));
    return OklaStatus::OKLA_STATUS_SUCCESS;
}
}  // namespace okla
}  // namespace matrix_ops