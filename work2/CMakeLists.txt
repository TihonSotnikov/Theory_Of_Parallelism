cmake_minimum_required(VERSION 3.10)
project(task1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-fopenmp -O3 -march=native)

add_executable(benchmark benchmark_all.cpp)
target_link_options(benchmark PRIVATE -fopenmp)
