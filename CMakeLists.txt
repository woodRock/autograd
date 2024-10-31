cmake_minimum_required(VERSION 3.18)

# Set C++ compiler before the project declaration
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)  # Explicitly set CUDA host compiler
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_C_COMPILER /usr/bin/gcc-13)

project(tensor_lib CUDA CXX)

# CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust for your GPU

# Add your source files
add_library(tensor_lib
    src/tensor_cuda.cu
)

target_include_directories(tensor_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(tensor_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(tensor_test src/tensor.cpp)
target_link_libraries(tensor_test tensor_lib)