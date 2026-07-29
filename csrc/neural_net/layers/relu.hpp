#pragma once

#include "./layer.hpp"
#include "../../tensor/tensor.hpp"

template <typename T>
class ReLU : public Layer<T>{

    private:

    public:
        ReLU()
        : Layer<T>(Tensor<T>({1}, T(0)),
                   Tensor<T>({1}, T(0)),
                   Tensor<T>({1}, T(0)),
                   false)
        {}

        Tensor<T> forward(const Tensor<T>& x) override {
            this->lastInput_ = x;
            Tensor<T> t = x;
            for(auto v : t.data()){
                if(v < 0){v = 0;} 
            }
            return t;
        }

        Tensor<T> backward(const Tensor<T>& lGrad) override {
            Tensor<T> xGrad = lGrad;
            const auto& in = this->lastInput_.data();
            auto& g = xGrad.data();
            for (size_t i = 0; i < g.size(); ++i) {
                if (in[i] <= T(0)) g[i] = T(0);
            }
            return xGrad;
        }

};