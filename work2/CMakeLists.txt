cmake_minimum_required(VERSION 3.10)
project(task1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-fopenmp -O3 -march=native)

add_executable(benchmark_static benchmark_all.cpp)
target_link_options(benchmark_static PRIVATE -fopenmp)

add_executable(benchmark_dynamic benchmark_all.cpp)
target_link_options(benchmark_dynamic PRIVATE -fopenmp)
target_compile_definitions(benchmark_dynamic PRIVATE schedule_dynamic)

add_executable(benchmark_guided benchmark_all.cpp)
target_link_options(benchmark_guided PRIVATE -fopenmp)
target_compile_definitions(benchmark_guided PRIVATE schedule_guided)
