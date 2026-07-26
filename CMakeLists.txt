cmake_minimum_required(VERSION 3.15)
project(ml_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Python_ROOT_DIR "${CMAKE_SOURCE_DIR}/.venv" CACHE PATH "" FORCE)
set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_REGISTRY NEVER)
set(CMAKE_FIND_FRAMEWORK NEVER)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python: ${Python_EXECUTABLE}")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_DIR_RESULT
)

if(NOT pybind11_DIR_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not run 'python -m pybind11 --cmakedir' using ${Python_EXECUTABLE}. "
        "Try: \"${Python_EXECUTABLE}\" -m pip install pybind11"
    )
endif()

message(STATUS "pybind11_DIR: ${pybind11_DIR}")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tensor
    framework/csrc/pybind/tensor_bind.cpp
)

target_include_directories(tensor PRIVATE
    framework/csrc
)

if(MINGW)
    target_link_options(tensor PRIVATE
        -static-libgcc -static-libstdc++
        -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive
    )
endif()