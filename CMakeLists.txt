cmake_minimum_required (VERSION 3.10)

set(CMAKE_CUDA_HOST_COMPILER "$ENV{CU_CCBIN}" CACHE PATH "$ENV{CU_CCBIN}" )
project(FinalProject LANGUAGES C CXX CUDA)
set(CMAKE_BUILD_TYPE, Release)
set(CMAKE_C_FLAGS "-Wall -Wextra -O3 -lm")

add_executable(collide collide.cu)
add_executable(collide_bf collide_bf.c)

