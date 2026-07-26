#include <pybind11/pybind11.h>
#include "../neural_net/neural_net.hpp"

namespace py = pybind11;

PYBIND11_MODULE(neural_net, m) {
    py::class_<Linear>(m, "Linear")
        .def(py::init<size_t, size_t, bool>(), py::arg("inFeats"), py::arg("outFeats"), py::arg("bias") = true)
        .def("forward", &Linear::forward, py::arg("x"))
        .def("backward", &Linear::backward, py::arg("lGrad"));
}