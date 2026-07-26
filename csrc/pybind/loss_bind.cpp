#include <pybind11/pybind11.h>
#include "../loss/mse.hpp"

namespace py = pybind11;

PYBIND11_MODULE(loss, m) {
    py::class_<MSE>(m, "MSE")
        .def(py::init<>())
        .def("forward", &MSE::forward, py::arg("pred"), py::arg("target"))
        .def("backward", &MSE::backward, py::arg("pred"), py::arg("target"));
}