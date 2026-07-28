#pragma once

#include <vector>
#include "../tensor/tensor.hpp"
#include "./layers.hpp"

template <typename T>
class Model{
    protected:
        bool trainingMode_;
        std::vector<Layer<T>*> layers_;

        Model() : trainingMode_(false){}

    public:

        virtual ~Model() = default;

        void train(){this->trainingMode_ = true}
        void eval(){this->trainingMode_ = false}
        bool isTraining(){return this->trainingMode_;}
        void addLayer(Layer<T>& layer){this->layers_.push_back(&layer)}
        virtual Tensor<T> forward(const Tensor<T>& x);

        Tensor<T> backward(const Tensor<T>& lGrad){
            Tensor<T> grad = lGrad;
            for(auto i = layers_.rbegin(); i != layers_.rend(); ++i){
                grad = (*i)->backward(grad);
            }
            return grad;
        }

        void step(const Tensor<T>& x, const Tensor<T>& yTrue){
            Tensor<T> yPred = forward(x);
            Tensor<T> lGrad = Tensor<T>::subtract(yPred, yTrue);
            backward(lGrad);
        }
};