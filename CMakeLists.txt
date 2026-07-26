cmake_minimum_required(VERSION 3.15)
project(ml_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force CMake to look at the venv's Python FIRST, before anything on PATH
# (e.g. MSYS2's python.exe, which doesn't have pybind11 installed).
set(Python_ROOT_DIR "${CMAKE_SOURCE_DIR}/.venv" CACHE PATH "" FORCE)
set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_REGISTRY NEVER)          # ignore any Python found via Windows registry
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

pybind11_add_module(_dlf
    csrc/pybind/bind.cpp
    csrc/pybind/tensor_bind.hpp
    csrc/pybind/loss_bind.hpp
    csrc/pybind/neural_net_bind.hpp
)

target_include_directories(_dlf PRIVATE
    csrc
)

# Output the compiled extension directly inside the python/dlf package
# folder, alongside __init__.py, so `import dlf` finds everything together.
set_target_properties(_dlf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dlf_pkg
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dlf_pkg
)

# Statically link the MinGW runtime so the built .pyd doesn't depend on
# MSYS2's DLLs (libstdc++-6.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll)
# being discoverable on PATH at import time.
if(MINGW)
    target_link_options(_dlf PRIVATE
        -static-libgcc -static-libstdc++
        -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive
    )
endif()