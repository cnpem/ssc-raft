cmake_minimum_required(VERSION 3.18)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(FetchContent)

# Order matter on the CMake file!1
# 2026-04-07 Necessary for mafalda-2. Needs to be first!
set(CMAKE_CUDA_HOST_COMPILER $ENV{CC})

# 2026-04-07 Change the order of this 2 commands, so it can be compiled in mafalda-2
set(CMAKE_CUDA_ARCHITECTURES 70)
project(sscRaft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) #-fPIC
add_compile_options(-use_fast_math)
# add_compile_options(-use_fast_math -g -G -Xcompiler -rdynamic -lineinfo)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

include_directories(
    cuda/inc/
    cuda/inc/common/
    cuda/inc/common10/
)

file(GLOB_RECURSE RAFT_SRC cuda/src/**/*.c*)

add_library(raft SHARED ${RAFT_SRC})

# ----- Add Cuda library path -------------------------------------
message(STATUS "CUDA lib dir: ${CUDAToolkit_LIBRARY_DIR}")
set(CUDA_LIB_DIR ${CUDAToolkit_LIBRARY_DIR})

set_target_properties(raft PROPERTIES
    INSTALL_RPATH "${CUDA_LIB_DIR}"
)
# -----------------------------------------------------------------

target_link_libraries(raft
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
)

install(TARGETS raft LIBRARY DESTINATION ${PROJECT_NAME}/lib)

