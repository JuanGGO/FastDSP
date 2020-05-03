cmake_minimum_required(VERSION 3.16)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)

include_directories(../cuda/structures/include
                        ${CUDA_INCLUDE_DIRS})
        #/usr/local/cuda-10.2/include)

link_directories(../cuda/cmake-build-release/structures/src/)
                    #/usr/local/cuda-10.2/lib64)

set(EXTRA_LIBS ${EXTRA_LIBS} fdspstructs)
add_executable(cpp main.cpp)

target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS} ${CUDA_LIBRARIES})