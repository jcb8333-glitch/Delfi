#pragma once

#include <vector>
#include "../tensor/tensor.hpp"
#include "./layers/layer.hpp"

template <typename T>
class Module : public Layer{
    protected:
        bool trainingMode_;
        std::vector<Layer<T>*> children_;

        Module() : Layer<T>(
            Tensor<T>({1}, T(0)),
            Tensor<T>({1}, T(0)),
            Tensor<T>({1}, T(0)),
            false
        ),
        trainingMode_(false){}

    public:

        virtual ~Module() = default;

        void train(){
            trainingMode_ = true;
            for(auto* child : children_){
                if(auto* m = dynamic_cast<Module<T>*>(child){
                    m->train();
                }
            }
        }

        void eval(){
            trainingMode_ = false;
            for(auto* child : children_){
                if(auto* m = dynamic_cast<Module<T>*>(child)){
                    m-eval();
                }
            }
        }

        bool isTraining(){return this->trainingMode_;}

        void addLayer(Layer<T>& layer){this->children_.push_back(&layer);}
        void addSubmodule(Module<T>& submod){this->submods_.push_back(&submod);}

        virtual Tensor<T> forward(const Tensor<T>& x) = 0;

        Tensor<T> backward(const Tensor<T>& lGrad){
            Tensor<T> grad = lGrad;
            for(auto i = children_.rbegin(); i != children_.rend(); ++i){
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