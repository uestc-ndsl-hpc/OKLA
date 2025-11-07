cmake_minimum_required(VERSION 3.22)

project(okla LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
)

FetchContent_Declare(
    argh
    GIT_REPOSITORY https://github.com/adishavit/argh.git
    GIT_TAG v1.3.2
)


FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(fmt argh googletest)

find_package(CUDAToolkit REQUIRED)

find_package(mathdx REQUIRED)

# 收集 matrix_ops 目录下的所有 .cu 文件
file(GLOB MATRIX_OPS_SOURCES "${CMAKE_SOURCE_DIR}/src/matrix_ops/*.cu")
    
add_library(matrix_ops_lib ${MATRIX_OPS_SOURCES})

target_link_libraries(matrix_ops_lib PUBLIC
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    mathdx::cusolverdx
    mathdx::cusolverdx_fatbin
    argh
    fmt::fmt
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON
    CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set_target_properties(matrix_ops_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(okla-test src/main.cu)

target_link_libraries(
    okla-test PRIVATE
    matrix_ops_lib
)

target_compile_options(okla-test PRIVATE "--extended-lambda")

# Ensure okla-test uses separable compilation and pass -dlto for nvlink.
set_target_properties(okla-test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON)

enable_testing()
add_subdirectory(tests)
