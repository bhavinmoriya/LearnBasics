cmake_minimum_required(VERSION 3.16)

# Project declaration
project(MyProject
    VERSION 1.0.0
    DESCRIPTION "My first CMake project"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add executable target
add_executable(my_app
    src/main.cpp
    src/math_utils.cpp
)

# Add include directory
target_include_directories(my_app PRIVATE src)
