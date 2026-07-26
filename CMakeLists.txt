cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(ml_framework LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tensor
    framework/csrc/pybind/tensor_bind.cpp
)

target_include_directories(tensor PRIVATE 
    framework/csrc
)