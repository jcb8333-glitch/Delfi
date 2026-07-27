#pragma once

#include <vector>
#include "../tensor/tensor.hpp"

template <typename T>
class Layer {
    protected:
        Tensor<T> lastInput_;
        Tensor<T> weights_;
        Tensor<T> bias_;
        T learningRate_ = T(0.01);
        bool useBias_;

        Layer(Tensor<T> lastInput, Tensor<T> weights, Tensor<T> bias, bool useBias)
            : lastInput_(std::move(lastInput)),
              weights_(std::move(weights)),
              bias_(std::move(bias)),
              useBias_(std::move(useBias))
            {}
    
        void updateWeights(const Tensor<T>& wGrad){
            auto& w = weights_.data();
            const auto& grad = wGrad.data();
            for(size_t i = 0; i < w.size(); ++i){
                w[i] -= learningRate_ * grad[i];
            }
        }
        void updateBias(const Tensor<T> bGrad){
            if(!useBias_) return;
            auto& b = bias_.data();
            const auto& grad = bGrad.data();
            for(size_t i; i < b.size(); ++i){
                b[i] = learningRate_ * grad[i];
            }
        }

        public:
            virtual Tensor<T> forward(const Tensor<T>& x) = 0;
            virtual Tensor<T> backward(const Tensor<T>& lGrad) = 0;
            virtual ~Layer() = default;
};

template <typename T>
class Linear : public Layer<T> {

    public:
        Linear : Layer<T>({1}, {inFeats}, T(0.5));

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

            Layer::updateWeights(wGrad);
            Layer::updateBias(bGrad);
        }

};