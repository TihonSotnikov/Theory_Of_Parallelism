cmake_minimum_required(VERSION 3.10)
project(task1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DOUBLE "Use double precision instead of float" ON)

add_executable(task1 1.cpp)

if (USE_DOUBLE)
    target_compile_definitions(task1 PRIVATE USE_DOUBLE)
endif()