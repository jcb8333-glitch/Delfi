#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../tensor/tensor.hpp"
#include "../tensor/dtype.hpp"

namespace py = pybind11;

void bind_tensor(py::module_& m) {
    py::class_<Tensor<float>>(m, "Tensor")
        .def(py::init<std::vector<size_t>, const float&>(), py::arg("shape"), py::arg("initial_val") = 0.0f)
        .def(py::init<std::vector<float>>(), py::arg("list_data"))
        .def(py::init<std::vector<std::vector<float>>>(), py::arg("list_data"))
        .def("data", static_cast<std::vector<float>& (Tensor<float>::*)()>(&Tensor<float>::data))
        .def("size", &Tensor<float>::size)
        .def("shape", &Tensor<float>::shape)
        .def("reshape", &Tensor<float>::reshape, py::arg("new_shape"))
        .def_static("transpose", &Tensor<float>::transpose)
        .def_static("multiply", &Tensor<float>::multiply)
        .def_static("subtract", &Tensor<float>::subtract)
        .def("__str__", &Tensor<float>::toString);
}