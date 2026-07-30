#include <pybind11/pybind11.h>
#include "../neural_net/layers/layer.hpp"
#include "../neural_net/layers/linear.hpp"
#include "../neural_net/layers/relu.hpp"
#include "../neural_net/layers/sigmoid.hpp"
#include "../neural_net/model.hpp"
#include "../tensor/tensor.hpp"

namespace py = pybind11;

class PyModel : public Model<float> {
    public:
        using Model<float>::Model;

        Tensor<float> forward(const Tensor<float>& x) override {
            PYBIND11_OVERRIDE_PURE(
                Tensor<float>,
                Model<float>,
                forward,
                x
            );
        }
};

void bind_neural_net(py::module_& m) {

    py::class_<Layer<float>>(m, "Layer");

    py::class_<Model<float>, PyModel, std::shared_ptr<Model<float>>>(m, "Model")
        .def(py::init<>())
        .def("train", &Model<float>::train)
        .def("eval", &Model<float>::eval)
        .def("is_training", &Model<float>::isTraining)
        .def("add_layer", &Model<float>::addLayer, py::arg("layer"), py::keep_alive<1,2>())
        .def("forward", &Model<float>::forward, py::arg("x"))
        .def("backward", &Model<float>::backward, py::arg("l_grad"))
        .def("step", &Model<float>::step, py::arg("x"), py::arg("y_true"));

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