#include <pybind11/pybind11.h>
#include "../neural_net/layers/layer.hpp"
#include "../neural_net/layers/linear.hpp"
#include "../neural_net/layers/relu.hpp"
#include "../neural_net/layers/sigmoid.hpp"
#include "../neural_net/module.hpp"
#include "../tensor/tensor.hpp"

namespace py = pybind11;

class PyModel : public Module<float> {
    public:
        using Module<float>::Module;

        Tensor<float> forward(const Tensor<float>& x) override {
            PYBIND11_OVERRIDE_PURE(
                Tensor<float>,
                Module<float>,
                forward,
                x
            );
        }
};

void bind_neural_net(py::module_& m) {

    py::class_<Layer<float>>(m, "Layer");

    py::class_<Module<float>, PyModel, std::shared_ptr<Module<float>>>(m, "Module")
        .def(py::init<>())
        .def("train", &Module<float>::train)
        .def("eval", &Module<float>::eval)
        .def("is_training", &Module<float>::isTraining)
        .def("add_layer", &Module<float>::addLayer, py::arg("layer"), py::keep_alive<1,2>())
        .def("forward", &Module<float>::forward, py::arg("x"))
        .def("backward", &Module<float>::backward, py::arg("l_grad"))
        .def("step", &Module<float>::step, py::arg("x"), py::arg("y_true"));

    py::class_<Linear<float>, Layer<float>>(m, "Linear")
        .def(py::init<size_t, size_t, bool>(), py::arg("inFeats"), py::arg("outFeats"), py::arg("bias") = true)
        .def("forward", &Linear<float>::forward, py::arg("x"))
        .def("backward", &Linear<float>::backward, py::arg("lGrad"));

    py::class_<ReLU<float>, Layer<float>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU<float>::forward, py::arg("x"))
        .def("backward", &ReLU<float>::backward, py::arg("lGrad"));

    py::class_<Sigmoid<float>, Layer<float>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid<float>::forward, py::arg("x"))
        .def("backward", &Sigmoid<float>::backward, py::arg("lGrad"));
    
};