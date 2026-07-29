#pragma once

#include "../../tensor/tensor.hpp"

template <typename T>
static Tensor<T> glorot_lecun(size_t outFeats, size_t inFeats){
            Tensor<T> w({outFeats, inFeats});
            auto& data = w.data();
            static thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            T scale = static_cast<T>(std::sqrt(1.0 / static_cast<double>(inFeats)));
            for (auto& v : data) v = static_cast<T>(dist(rng)) * scale;
            return w;
        }