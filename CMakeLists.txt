cmake_minimum_required(VERSION 3.17)
project(MatrixMul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Find Armadillo (installed via libarmadillo-dev)
find_package(Armadillo REQUIRED)

# -----------------------------
# Main executable: matrix_mul
# -----------------------------
add_executable(matrix_mul
    src/main.cpp
    src/MatrixMul.cpp
    src/gpu/matrixMul.cu
)

# Enable separable compilation for device functions
set_target_properties(matrix_mul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matrix_mul PRIVATE ${CMAKE_SOURCE_DIR}/src ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(matrix_mul PRIVATE ${ARMADILLO_LIBRARIES})

# -----------------------------
# Tests: test_matrix_mul
# -----------------------------
enable_testing()

add_executable(test_matrix_mul
    tests/test_matrix_mul.cpp
    src/MatrixMul.cpp
    src/gpu/matrixMul.cu
)

# Enable CUDA separable compilation for test
set_target_properties(test_matrix_mul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_matrix_mul PRIVATE ${CMAKE_SOURCE_DIR}/src ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(test_matrix_mul PRIVATE gtest_main ${ARMADILLO_LIBRARIES})

# Link to installed GoogleTest libraries
target_link_libraries(test_matrix_mul
 PRIVATE
    gtest
    gtest_main
    ${ARMADILLO_LIBRARIES}
)

# Register test
add_test(NAME MatrixMulTest COMMAND test_matrix_mul)

