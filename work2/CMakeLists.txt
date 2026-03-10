cmake_minimum_required(VERSION 3.16)
project(work2)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O2)

foreach(task task1 task2 task3)
    add_executable(${task} ${task}.cpp)
    target_link_libraries(${task} OpenMP::OpenMP_CXX)
endforeach()