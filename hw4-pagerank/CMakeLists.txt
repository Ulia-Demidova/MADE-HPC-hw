cmake_minimum_required(VERSION 3.17)
project(hw4 C)

set(CMAKE_C_STANDARD 99)

set(INC_DIR /usr/local/opt/openblas/include)
set(LINK_DIR /usr/local/opt/openblas/lib)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3")

add_executable(hw4 pagerank.c)
target_link_libraries(hw4 libopenblas.a)