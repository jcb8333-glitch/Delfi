#include <pybind11/pybind11.h>
#include "../cuda/accelerator.hpp"

namespace py = pybind11;

void bind_accelerator(py::module_& m){
    py::class_<Accelerator>(m, "Accelerator")
        .def_static("available", &Accelerator::available)
        .def_static("default_device", &Accelerator::defaultDevice);
}