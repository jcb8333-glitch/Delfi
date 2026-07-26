#include <pybind11/pybind11.h>
#include "../loss/mse.hpp"

namespace py = pybind11;

PYBIND11_MODULE(loss, m) {
    py::class_<MSEloss>(m, "MSEloss")
        .def(py::init<>())
        .def("forward", &MSEloss::forward, py::arg("pred"), py::arg("target"))
        .def("backward", &MSEloss::backward, py::arg("pred"), py::arg("target"));
}