#pragma once

#include <vector>
#include <type_traits>
#include <string>
#include <ostream>
#include <algorithm>

template <typename T>
class Tensor {
    private:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        std::vector<T> data_;

        void computeStrides() {
            strides_.resize(shape_.size());
            size_t acc = 1;
            for(size_t i = shape_.size(); i-- > 0;){
                strides_[i] = acc;
                acc *= shape_[i];
            }
        }

        std::string dimParse(size_t dim, size_t offset, size_t stride) const{
            std::string out = "[";
            size_t dimSize = shape_[dim];
            size_t innerStride = stride / dimSize;

            for (size_t i = 0; i < dimSize; ++i){
                if (dim == shape_.size() - 1){
                    out += std::to_string(data_[offset + i]);
                } else {
                    out += dimParse(dim + 1, offset + i * innerStride, innerStride);
                }
                if (i + 1 < dimSize) out += ", ";
            }
            out += "]";
            return out;
        }

        void collect(const T& val, std::vector<size_t>& outShape, std::vector<T>& outData, size_t depth){
            outData.push_back(val);
        }
        template <typename U>
        void collect(const std::vector<U>& val, std::vector<size_t>& outShape, std::vector<T>& outData, size_t depth){
            if (outShape.size() <= depth){
                outShape.push_back(val.size());
            } else if (outShape[depth] != val.size()) {
                throw std::invalid_argument("Tensor: Uneven dimensions in input vector");
            }
            unwrap(val, outShape, outData, depth);
        }

        template <typename U>
        void unwrap(const std::vector<U>& val, std::vector<size_t>& outShape, std::vector<T>& outData, size_t depth){
            for(const auto& item : val){
                collect(item, outShape, outData, depth +1);
            }
        }

    public:
        Tensor(std::vector<size_t> shape, const T& initial_val = T()) : shape_(std::move(shape)){
            size_t size = 1;
            for(auto dim : shape_) size *= dim;
            data_.assign(size, initial_val);
            computeStrides();
        }

        template <typename U>
        Tensor(const std::vector<U> list){
            std::vector<size_t> outShape;
            std::vector<T> outData;
            collect(list, outShape, outData, 0);

            shape_ = outShape;
            data_ = outData;
            computeStrides();
        }

        static Tensor<T> clone(const Tensor<T>& t){
            return t;
        }

        std::vector<T>& data() {
            return data_;
        }
        
        const std::vector<T>& data() const {
            return data_;
        }

        size_t size() const {
            return data_.size();
        }

        const std::vector<size_t>& shape() const {
            return shape_;
        }

        Tensor<T> reshape(const std::vector<size_t>& newShape) const{
            size_t  newSize = 1;
            for(auto dim : newShape) newSize *= dim;

            if (newSize != data_.size()) throw std::invalid_argument("Reshape: total element count must match new size");
            Tensor<T> result(newShape);
            result.data() = data_;
            return result;
        }

        static Tensor<T> transpose(const Tensor<T>& t){
            if (t.shape().size() == 1) {
                // Promote {n} directly to a column {n, 1}
                return t.reshape({t.shape()[0], 1});
            }

            std::vector<size_t> newShape = t.shape();
            std::reverse(newShape.begin(), newShape.end());
            Tensor<T> transposed(newShape);

            for (size_t i = 0; i < t.size(); ++i){
                std::vector<size_t> indices(t.shape().size());
                size_t offset = i;

                for (size_t j = 0; j < t.shape().size(); ++j){
                    indices[j] = offset / t.strides_[j];
                    offset %= t.strides_[j];
                }

                std::reverse(indices.begin(), indices.end());
                size_t newOffset = 0;

                for (size_t j = 0; j < indices.size(); ++j){
                    newOffset += indices[j] * transposed.strides_[j];
                }

                transposed.data()[newOffset] = t.data()[i];
            }
            return transposed;
        }

        static Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b){
            const auto& aShape = a.shape();
            const auto& bShape = b.shape();

            if ((aShape.size() != 1 && aShape.size() != 2) ||
                (bShape.size() != 1 && bShape.size() != 2))
                throw std::invalid_argument("matmul supports only 1D or 2D tensors");

            bool aWasVector = (aShape.size() == 1);
            bool bWasVector = (bShape.size() == 1);

            Tensor<T> aPromoted = aWasVector ? a.reshape({1, aShape[0]}) : a;
            Tensor<T> bPromoted = bWasVector ? b.reshape({bShape[0], 1}) : b;

            const auto& aShape2D = aPromoted.shape();
            const auto& bShape2D = bPromoted.shape();

            if (aShape2D[1] != bShape2D[0])
                throw std::invalid_argument("Incompatible shapes for multiplication");

            size_t M = aShape2D[0];
            size_t K = aShape2D[1];
            size_t N = bShape2D[1];

            Tensor<T> result({M, N});

            for (size_t i = 0; i < M; ++i){
                for (size_t j = 0; j < N; ++j){
                    T sum = T();
                    for (size_t k = 0; k < K; ++k){
                        sum += aPromoted(i, k) * bPromoted(k, j);
                    }
                    result(i, j) = sum;
                }
            }

            if (aWasVector && bWasVector) {
                return result.reshape({});
            } else if (aWasVector) {
                return result.reshape({N});
            } else if (bWasVector) {
                return result.reshape({M});
            }
            return result;
        }

        static Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b){
            if (a.shape() != b.shape()) throw std::invalid_argument("Subtract: Tensors are not the same shape");

            std::vector<T> aData = a.data();
            std::vector<T> bData = b.data();

            Tensor<T> result(a.shape());

            for(size_t i = 0; i < aData.size(); ++i){
                result.data()[i] = aData[i] - bData[i];
            }

            return result;
        }

        static Tensor<T> mean(const Tensor<T>& t){
            Tensor<T> result = Tensor<T>({1},0);
            const std::vector<T>& data = t.data();

            for(size_t i = 0; i < data.size(); ++i){
                result(0) += data[i];
            }
            result(0) /= t.size();
            return result;
        }

        template <typename... Idx>
        T& operator()(Idx... idx) {
            std::vector<size_t> indices{static_cast<size_t>(idx)...};
            size_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i){
                offset += indices[i] * strides_[i];
            }
            return data_[offset];
        }

        template <typename... Idx>
        const T& operator()(Idx... idx) const {
            std::vector<size_t> indices{static_cast<size_t>(idx)...};
            size_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i){
                offset += indices[i] * strides_[i];
            }
            return data_[offset];
        }

        std::string toString() const {
            if (shape_.empty()) return "[]";
            return dimParse(0, 0, data_.size());
        }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
    os << "Tensor(" << t.toString() << ")";
    return os;
}