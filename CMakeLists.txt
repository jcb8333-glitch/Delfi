cmake_minimum_required(VERSION 3.15)
project(ml_framework LANGUAGES CXX)

option(DLF_CUDA "Build with CUDA support" ON)
if(DLF_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "CUDA requested but not found")
        set(DLF_CUDA OFF)
    endif()
endif()

set(CMAKE_CXX_STANDARD 23)
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

pybind11_add_module(_delfi
    csrc/pybind/bind.cpp
    csrc/pybind/tensor_bind.hpp
    csrc/cuda/tensor_kernels.cu
    csrc/pybind/loss_bind.hpp
    csrc/pybind/neural_net_bind.hpp
    csrc/pybind/accelerator_bind.hpp
)

if(DLF_CUDA)
    target_compile_definitions(_delfi PRIVATE DLF_CUDA)
    target_link_libraries(_delfi PRIVATE CUDA::cudart)
endif()

target_include_directories(_delfi PRIVATE
    csrc
)

set_target_properties(_delfi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/delfi
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/delfi
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/delfi
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/delfi
)

if(MINGW)
    target_link_options(_delfi PRIVATE
        -static-libgcc -static-libstdc++
        -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive
    )
endif()