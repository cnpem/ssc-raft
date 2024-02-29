cmake_minimum_required(VERSION 3.11)

# include(FetchContent)

project(sscRaft LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) #-fPIC
add_compile_options(-use_fast_math)

# FetchContent_Declare(
#   ssccommons
#   GIT_REPOSITORY https://gitlab.cnpem.br/GCC/ssc-commons.git
#   GIT_TAG v0.0.1
# )
# FetchContent_MakeAvailable(ssccommons)

#toggle next lines to work in a local copy of ssc commons
# add_subdirectory(../ssc-commons ./_deps/ssccommons)
# set(ssccommons_SOURCE_DIR ../ssc-commons)

include_directories(
    cuda/inc
    cuda/inc/common10
)
# ${ssccommons_SOURCE_DIR}/cuda/inc/common

file(GLOB_RECURSE RAFT_SRC cuda/src/**/*.c*)
message("${RAFT_SRC}")
add_library(raft SHARED ${RAFT_SRC})

# target_link_libraries(raft ${CUDA_LIBRARIES} cufft cublas ssccommons_shared)
target_link_libraries(raft ${CUDA_LIBRARIES} cufft cublas)


# install(TARGETS ssccommons_shared LIBRARY DESTINATION ${PROJECT_NAME}/lib)
install(TARGETS raft LIBRARY DESTINATION ${PROJECT_NAME}/lib)

# set_target_properties(raft PROPERTIES INSTALL_RPATH "$ORIGIN") # look for binaries in same path (to link sscommons)
