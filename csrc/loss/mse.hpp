#pragma once

#include "../tensor/tensor.hpp"
#include <cmath>


template <typename T>
class MSEloss{
    private:
        Tensor<T> lastDiff_;
    public:
        MSEloss() : lastDiff_({1}, T(0)){}

        Tensor<T> forward(const Tensor<T>& actual, const Tensor<T>& prediction){
            Tensor<T> diff = Tensor<T>::subtract(prediction, actual);

            Tensor<T> square = diff;
            auto& data = square.data();
            for(size_t i = 0; i < data.size(); ++i){
                data[i] = data[i] * data[i];
            }

            lastDiff_ = diff;
            return Tensor<T>::mean(square, false);
        }

        Tensor<T> backward(){
            Tensor<T> grad = lastDiff_;
            auto& data = grad.data();
            T scale = T(2) / T(lastDiff_.size());

            for (size_t i = 0; i < data.size(); ++i){
                data[i] *= scale;
            }

            return grad;
        }
};