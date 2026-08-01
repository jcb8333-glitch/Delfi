
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

            for(size_t b = 0; b < batch; ++b){
                size_t base = b * feats;
                T maxVal = data[base];
                for(size_t i = 1; i < feats; ++i){
                    maxVal = std::max(maxVal, data[base + j]);
                }
                T sum = 0;
                for(size_t j = 0; j < feats; ++j){
                    data[base + j] = std::exp(data[base + j]-maxVal);
                    sum += data[base + j];
                }
                for (size_t i = 0; i < feats; ++i){
                    data[base + i] /= sum;
                }
            }
            this->lastInput_ = y;
            return y;
        }
        
        Tensor<T> backward(const Tensor<T>& lGrad) override {}

};