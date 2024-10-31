cmake_minimum_required(VERSION 3.18)
project(tensor_project CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add OpenMP flags to CUDA compilation
if(OpenMP_CXX_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

# Set source files
set(SOURCES
    src/tensor.cpp
    src/tensor_cuda.cu
)

# Create executable
add_executable(output ${SOURCES})

# Set CUDA architecture - adjust based on your GPU
set_property(TARGET output PROPERTY CUDA_ARCHITECTURES 60 70 75)

# Set properties for the target
set_target_properties(output PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(output PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optional: Add optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(output PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
    )
endif()