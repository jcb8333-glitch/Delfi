#include <pybind11/pybind11.h>
#include "tensor_bind.hpp"
#include "loss_bind.hpp"
#include "neural_net_bind.hpp"
#include "accelerator_bind.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_delfi, m) {
    bind_tensor(m);
    bind_loss(m);
    bind_neural_net(m);
    bind_accelerator(m);
}