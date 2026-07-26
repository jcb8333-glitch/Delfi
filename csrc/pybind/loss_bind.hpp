#include <pybind11/pybind11.h>
#include "../loss/mse.hpp"

namespace py = pybind11;

void bind_loss(py::module_& m) {
    py::class_<MSEloss<float>>(m, "MSEloss")
        .def(py::init<>())
        .def("forward", &MSEloss<float>::forward, py::arg("target"), py::arg("pred"))
        .def("backward", &MSEloss<float>::backward);
}