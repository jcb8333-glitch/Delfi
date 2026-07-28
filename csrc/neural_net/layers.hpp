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
            for(size_t i = 0; i < b.size(); ++i){
                b[i] -= learningRate_ * grad[i];
            }
        }

        public:
            virtual Tensor<T> forward(const Tensor<T>& x) = 0;
            virtual Tensor<T> backward(const Tensor<T>& lGrad) = 0;
            virtual ~Layer() = default;
};

template <typename T>
class Linear : public Layer<T> {
    private:
        size_t inFeats_;
        size_t outFeats_;

    public:
        Linear(size_t inFeats, size_t outFeats, bool bias=true)
        : Layer<T>(Tensor<T>({1, inFeats}, T(0)),
                   Tensor<T>({outFeats, inFeats}, T(0)),
                   Tensor<T>({outFeats}, T(0)),
                   bias),
            inFeats_(inFeats),
            outFeats_(outFeats) 
        {}


        Tensor<T> forward(const Tensor<T>& x) override {
            
            this->lastInput_ = x;

            Tensor<T> y = Tensor<T>::multiply(x, Tensor<T>::transpose(this->weights_));
            
            size_t batch = y.shape()[0];
            if(this->useBias_){
                for(size_t i = 0; i < batch; ++i){
                    for(size_t j = 0; j < outFeats_; ++j)
                        y(i, j) += this->bias_(j);
                }
            }
            return y;
        }

        Tensor<T> backward(const Tensor<T>& lGrad) override {
            const auto& gradShape = lGrad.shape();
            const auto& inputShape = this->lastInput_.shape();

            if (gradShape.size() != 2 || gradShape[1] != outFeats_)
                throw std::invalid_argument("Linear::backward: gradient shape mismatch");
            if (inputShape.size() != 2 || inputShape[1] != inFeats_ || inputShape[0] != gradShape[0])
                throw std::invalid_argument("Linear::backward: cached input shape mismatch");

            size_t batchSize = gradShape[0];

            Tensor<T> ones({batchSize}, T(1));
            Tensor<T> bGrad =
                Tensor<T>::multiply(
                    Tensor<T>::transpose(lGrad),
                    ones);

            Tensor<T> wGrad =
                Tensor<T>::multiply(
                    Tensor<T>::transpose(lGrad),
                    this->lastInput_);

            Tensor<T> xGrad =
                Tensor<T>::multiply(
                    lGrad,
                    this->weights_);

            this->updateWeights(wGrad);
            this->updateBias(bGrad);
            return xGrad;
        }

};