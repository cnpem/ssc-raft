cmake_minimum_required(VERSION 3.10)

project(sscRaft LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) #-fPIC
add_compile_options(-use_fast_math)

include_directories(
    cuda/inc
    cuda/inc/common10
)

file(GLOB_RECURSE RAFT_SRC cuda/src/**/*.c*)
message("${RAFT_SRC}")
add_library(raft SHARED ${RAFT_SRC})

target_link_libraries(raft ${CUDA_LIBRARIES} cufft cublas)

install(TARGETS raft LIBRARY DESTINATION ${PROJECT_NAME}/lib)
