#include <pybind11/pybind11.h>
#include "../neural_net/layers.hpp"

namespace py = pybind11;

void bind_neural_net(py::module_& m) {
    py::class_<Linear<float>>(m, "Linear")
        .def(py::init<size_t, size_t, bool>(), py::arg("inFeats"), py::arg("outFeats"), py::arg("bias") = true)
        .def("forward", &Linear<float>::forward, py::arg("x"))
        .def("backward", &Linear<float>::backward, py::arg("lGrad"));
}