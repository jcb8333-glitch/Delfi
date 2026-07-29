#include "./layer.hpp"
#include "../../tensor/tensor.hpp"

template <typename T>
class ReLU{

    public:
        ReLU(){}

        Tensor<T> forward(Tensor<T>& x){
            for(auto& v : x.data()){
                if(v < 0){v = 0;} 
            }
            return x
        }
};