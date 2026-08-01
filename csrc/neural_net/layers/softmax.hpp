#pragma once

#include <cmath>
#include "./layer.hpp"
#include "../../tensor/tensor.hpp"

template <typename T>
class Softmax : public Layer<T>{

    public:
        Softmax(size_t dims)
        : Layer<T>(Tensor<T>({1}, T(0)),
                   Tensor<T>({1}, T(0)),
                   Tensor<T>({1}, T(0)),
                   false)
        {}

        Tensor<T> forward(const Tensor<T>& x) override {
            Tensor<T> y = x;
            size_t batch = x.shape()[0];
            size_t feats = x.shape()[1];
            auto& data = y.data();

            for(auto b = 0uz; b < batch; ++b){
                size_t batchStart = b * feats;
                T maxVal = data[batchStart];
                for(auto i = 0uz; i < feats; ++i){
                    maxVal = std::max(maxVal, data[batchStart + j]);
                }
                T sum = 0;
                for(auto j = 0uz; j < feats; ++j){
                    data[batchStart + j] = std::exp(data[batchStart + j]-maxVal);
                    sum += data[batchStart + j];
                }
                for (auto i = 0uz; i < feats; ++i){
                    data[batchStart + i] /= sum;
                }
            }
            this->lastInput_ = y;
            return y;
        }
        
        Tensor<T> backward(const Tensor<T>& lGrad) override {
            Tensor<T> xGrad = (lGrad.shape(), T(0));
            auto& data = x.data();
            size_t batch = lGrad.shape()[0];
            size_t feats = lGrad.shape()[1];

            for(auto b = 0uz; b < batch; ++b){
                size_t base = b * feats;
                for(i = 0uz; i < batch; ++i){
                    T sum = T(0);
                    for(auto j = 0uz; j < feats; ++j){
                        T kronecker = (i==j) ? T(1) : 0;
                        sum += g[base + j] * a[base + j] * (kronecker - s[base + i]);
                    }
                    data[base+i] = sum;
                }
            }
            return xGrad;
        }
};