cmake_minimum_required(VERSION 3.16)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})

#initialization
include_directories(${PROJECT_SOURCE_DIR}/core/include)
add_subdirectory(core/src)

#structures
include_directories(${PROJECT_SOURCE_DIR}/structures/include)
add_subdirectory(structures/src)

set(EXTRA_LIBS ${EXTRA_LIBS} fdspstructs fdspinit )

add_executable(cuda main.cu)

target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS} cudart)

set_target_properties(
        cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)