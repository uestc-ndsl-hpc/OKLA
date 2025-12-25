add_executable(matrix_ops_tests cholesky_test.cu trsm_test.cu
               ${CMAKE_SOURCE_DIR}/src/okla_warppers/trsm_kernel.cu)

target_link_libraries(matrix_ops_tests PRIVATE matrix_ops_lib GTest::gtest_main)

target_link_options(matrix_ops_tests PRIVATE $<DEVICE_LINK:-dlto>)
target_compile_options(
  matrix_ops_tests
  PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>")

target_compile_options(matrix_ops_tests PRIVATE "--extended-lambda")

include(GoogleTest)
gtest_discover_tests(matrix_ops_tests)

set_target_properties(
  matrix_ops_tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                              INTERPROCEDURAL_OPTIMIZATION OFF)

add_executable(cutlass3test cutlass3test.cu)

set_target_properties(
  cutlass3test
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
             INTERPROCEDURAL_OPTIMIZATION OFF
             CUDA_ARCHITECTURES OFF # 禁用 CMake 的自动架构管理
)

target_link_libraries(cutlass3test PRIVATE matrix_ops_lib cutlass_headers)

add_executable(cutlass2test cutlass2test.cu)

set_target_properties(
  cutlass2test
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
             INTERPROCEDURAL_OPTIMIZATION OFF
             CUDA_ARCHITECTURES OFF # 禁用 CMake 的自动架构管理
)

target_link_libraries(cutlass2test PRIVATE cutlass_headers fmt::fmt)

add_executable(okla_trsm_bench
               okla_trsm_bench.cu
               ${CMAKE_SOURCE_DIR}/src/okla_warppers/trsm_kernel.cu)

set_target_properties(
  okla_trsm_bench
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             INTERPROCEDURAL_OPTIMIZATION OFF)

target_link_libraries(okla_trsm_bench PRIVATE CUDA::cudart)
