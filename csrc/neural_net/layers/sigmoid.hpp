#pragma once

#include <cmath>
#include "layer.hpp"
#include "weight_init.hpp"
#include "../../tensor/tensor.hpp"

template <typename T>
class Sigmoid : public Layer<T>{
    public:
        Sigmoid()
        : Layer<T>(Tensor<T>({1}, 0),
                   Tensor<T>({1}, 0),
                   Tensor<T>({1}, 0),
                   true)
        {}

        Tensor<T> forward(const Tensor<T>& x) override {
            this->lastInput_ = x;
            Tensor<T> y = x;
            auto& data = y.data();
            for(auto& v : data){
                v = T(1) / (T(1)+std::exp(-v));
            }
            return y;
        }

        Tensor<T> backward(const Tensor<T>& lGrad) override {
            Tensor<T> xGrad = lGrad;
            const auto& in = this->lastInput_.data();
            auto& g = xGrad.data();
            for(size_t i = 0; 0 < g.size(); ++i){
                T s = T(1) / T(1) + std::exp(-in[i]);
                g[i] *= s * (T(1) - s);
            }
            return xGrad;
        }

};