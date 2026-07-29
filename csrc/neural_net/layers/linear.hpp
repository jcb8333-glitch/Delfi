#pragma once

#include <random>
#include "./layer.hpp"
#include "../../tensor/tensor.hpp"

template <typename T>
class Linear : public Layer<T> {
    private:
        size_t inFeats_;
        size_t outFeats_;

        static Tensor<T> makeWeights(size_t outFeats, size_t inFeats){
            Tensor<T> w({outFeats, inFeats});
            auto& data = w.data();
            static thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            T scale = static_cast<T>(std::sqrt(1.0 / static_cast<double>(inFeats)));
            for (auto& v : data) v = static_cast<T>(dist(rng)) * scale;
            return w;
        }

    public:
        Linear(size_t inFeats, size_t outFeats, bool bias=true)
        : Layer<T>(Tensor<T>({1, inFeats}, T(0)),
                   makeWeights(outFeats, inFeats),
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
            T scale = T(1) / static_cast<T>(batchSize);

            Tensor<T> ones({batchSize}, T(1));

            Tensor<T> bGrad =
                Tensor<T>::multiply(
                    Tensor<T>::transpose(lGrad),
                    ones);

            Tensor<T> wGrad =
                Tensor<T>::multiply(
                    Tensor<T>::transpose(lGrad),
                    this->lastInput_);
            
            for (auto& v : bGrad.data()) v *= scale;
            for (auto& v : wGrad.data()) v *= scale;

            Tensor<T> xGrad =
                Tensor<T>::multiply(
                    lGrad,
                    this->weights_);

            this->updateWeights(wGrad);
            this->updateBias(bGrad);
            return xGrad;
        }

};