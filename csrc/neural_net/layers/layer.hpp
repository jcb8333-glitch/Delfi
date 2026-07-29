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