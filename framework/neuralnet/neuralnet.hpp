#pragma once

#include <vector>
#include "../tensor/tensor.hpp"
#include <iostream>

template <typename T>
class Linear {

    private:
        Tensor<T> lastInput_;
        Tensor<T> weights_;
        T bias_;

        void updateWeights(){}

        void updateBias(){}

    public:
        Linear(size_t inFeats, size_t outFeats, bool bias=true)
            : weights_({inFeats}, T(0.0)), bias_(T(0.5)) {
        }

        Tensor<T> forward(Tensor<T>& x){
            lastInput_ = x;
            Tensor<T> y = Tensor<T>::multiply(x, Tensor<T>::transpose(weights_));
            auto& data = y.data();
            for(size_t i = 0; i < y.size(); ++i){
                data[i] += bias_;
            }
            return y;
        }

        void backward(const Tensor<T>& lGrad){
            Tensor<T> wGrad = Tensor<T>({1},0);
            Tensor<T> bGrad = Tensor<T>({1},0);
            auto& data = lGrad.data();

            for(size_t i = 0; i < lGrad.size(); ++i){
                wGrad(0) += data[i] * lastInput_.data()[i];
            }
            for(size_t i = 0; i < lGrad.size(); ++i){
                bGrad(0) += data[i];
            }
            updateWeights(wGrad);
            updateBias(bGrad);
        }

};

template <typename T>
class ReLU {

    public:
        ReLU(){}

        Tensor<T> relu(Tensor<T> x){
            auto& data = x.data();
            for(size_t i = 0; i < x.size(); ++i){
                if(data[i] < 0) data[i] = 0;
            }
            return x;
        }

};