cmake_minimum_required(VERSION 3.22)
project(okla LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- 1. 建议将全局优化开关放在 target 创建之前 ---
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
# 如果你不需要极致的链接时优化，可以把下面这行关掉来消除警告
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# --- 2. 依赖管理 ---
include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.2.0)

FetchContent_Declare(
  argh
  GIT_REPOSITORY https://github.com/adishavit/argh.git
  GIT_TAG v1.3.2)

FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v4.3.3)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()

add_library(cutlass_headers INTERFACE)

target_include_directories(
  cutlass_headers
  INTERFACE ${cutlass_SOURCE_DIR}/include
            ${cutlass_SOURCE_DIR}/tools/util/include
            ${cutlass_SOURCE_DIR}/tools/library/include)

target_compile_features(cutlass_headers INTERFACE cxx_std_17)

FetchContent_MakeAvailable(fmt argh googletest)

find_package(CUDAToolkit REQUIRED)

# --- 3. 库定义 ---
file(GLOB MATRIX_OPS_SOURCES "${CMAKE_SOURCE_DIR}/src/matrix_ops/*.cu")
add_library(matrix_ops_lib STATIC ${MATRIX_OPS_SOURCES}) # 显式指定 STATIC

target_link_libraries(matrix_ops_lib PUBLIC CUDA::cublas CUDA::cusolver
                                            CUDA::curand argh fmt::fmt)

target_link_libraries(matrix_ops_lib PUBLIC cutlass_headers)

target_compile_options(
  matrix_ops_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info
                        --extended-lambda --expt-relaxed-constexpr>)

# --- 4. 可执行文件 ---
add_executable(okla-test src/main.cu)
target_link_libraries(okla-test PRIVATE matrix_ops_lib)

# 这样设置可以确保 okla-test 继承 lib 的编译选项
target_compile_options(okla-test
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

enable_testing()
add_subdirectory(tests)
