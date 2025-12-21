add_executable(matrix_ops_tests
    cholesky_test.cu
)

target_link_libraries(matrix_ops_tests PRIVATE
    matrix_ops_lib
    GTest::gtest_main
)

target_link_options(matrix_ops_tests PRIVATE $<DEVICE_LINK:-dlto>)
target_compile_options(matrix_ops_tests PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
)

target_compile_options(matrix_ops_tests PRIVATE "--extended-lambda")

include(GoogleTest)
gtest_discover_tests(matrix_ops_tests)

set_target_properties(matrix_ops_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON)

