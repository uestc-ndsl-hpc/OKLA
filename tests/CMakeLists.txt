add_executable(matrix_ops_tests
    cholesky_test.cu
)

target_link_libraries(matrix_ops_tests PRIVATE
    matrix_ops_lib
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matrix_ops_tests)
