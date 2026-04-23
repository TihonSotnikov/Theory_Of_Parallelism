cmake_minimum_required(VERSION 3.15)
project(ParallelBench LANGUAGES CXX)

option(ENABLE_LOGGING "Enable benchmark logging" ON)

add_executable(bench task1.cpp)

target_compile_features(bench PRIVATE cxx_std_20)

target_compile_options(bench PRIVATE 
    -O3 
    -march=native 
    -ffast-math
)

if(ENABLE_LOGGING)
    target_compile_definitions(bench PRIVATE BENCHMARK_LOGGING)
endif()

find_package(Threads REQUIRED)
target_link_libraries(bench PRIVATE Threads::Threads)
