#pragma once

#include <vector>
#include "../tensor/tensor.hpp"
#include <iostream>

template <typename T>
class Linear {

    private:
        Tensor<T> lastInput_;
        Tensor<T> weights_;
        T learningRate_ = T(0.01);
        T bias_;

        void updateWeights(const Tensor<T>& wGrad){}
            auto& w = weights_.data();
            const auto& grad = wGrad.data();
            for(size_t i = 0; i < w.size(); ++i){
                w[i] -= learningRate_ * g[i];
            }
        void updateBias(T b){
            bias_ -= learningRate_ * b;
        }

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
            size_t inFeats = weights_.size();
            size_t batchSize = lGrad.size();

            Tensor<T> wGrad = Tensor<T>({inFeats}, T(0));
            T bGrad = T(0);

            const auto& gradData = lGrad.data();
            const auto& inputData = lastInput_.data();
            auto& wGradData = wGrad.data();

            for (size_t i = 0; i < batchSize; ++i){
                T g = gradData[i];
                bGrad += g;

                size_t base = i * inFeats;
                for (size_t j = 0; j < inFeats; ++j){
                    wGradData[j] += g * inputData[base + j];
                }
            }

            updateWeights(wGrad);
            updateBias(bGrad);
        }

};