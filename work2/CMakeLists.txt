cmake_minimum_required(VERSION 3.10)
project(task1 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -march=native)

find_package(OpenMP REQUIRED)

add_executable(benchmark_static benchmark_all.cpp)
target_link_libraries(benchmark_static PRIVATE OpenMP::OpenMP_CXX)

add_executable(benchmark_dynamic benchmark_all.cpp)
target_link_libraries(benchmark_dynamic PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(benchmark_dynamic PRIVATE schedule_dynamic)

add_executable(benchmark_guided benchmark_all.cpp)
target_link_libraries(benchmark_guided PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(benchmark_guided PRIVATE schedule_guided)