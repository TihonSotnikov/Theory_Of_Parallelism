cmake_minimum_required(VERSION 3.15)
project(ParallelBench LANGUAGES CXX)

option(BENCHMARK_LOGGING "Enable benchmark logging" OFF)

add_executable(bench task1.cpp)
target_compile_features(bench PRIVATE cxx_std_20)

target_compile_options(bench PRIVATE -O3)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(bench PRIVATE 
        -march=native 
        -ffast-math
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(bench PRIVATE 
        -fast
    )
endif()

if(BENCHMARK_LOGGING)
    target_compile_definitions(bench PRIVATE BENCHMARK_LOGGING)
endif()

find_package(Threads REQUIRED)
target_link_libraries(bench PRIVATE Threads::Threads)
