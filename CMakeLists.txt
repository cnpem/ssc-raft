cmake_minimum_required(VERSION 3.10)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(FetchContent)

project(sscRaft LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) #-fPIC
# add_compile_options(-use_fast_math)
# add_compile_options(-use_fast_math -g -G -Xcompiler -rdynamic -lineinfo)
add_compile_options(-use_fast_math -g -G -Xcompiler -rdynamic)



find_package(CUDAToolkit)

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

target_link_libraries(raft ${CUDA_LIBRARIES} CUDA::cufft CUDA::cublas)

install(TARGETS raft LIBRARY DESTINATION ${PROJECT_NAME}/lib)

