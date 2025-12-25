#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace matrix_ops {
namespace okla {

template <typename T, cublasSideMode_t SideMode, cublasFillMode_t FillMode,
          cublasOperation_t Operation, cublasDiagType_t Diag, int M, int Nrhs>
__global__ void trsm_kernel(T alpha, const T* A, int lda, T* B, int ldb);

}  // namespace okla
}  // namespace matrix_ops
