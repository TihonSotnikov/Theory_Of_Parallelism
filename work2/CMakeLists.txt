cmake_minimum_required(VERSION 3.10)
project(task1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fopenmp)

add_executable(benchmark benchmark_all.cpp)