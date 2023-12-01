#pragma once

#include <chrono>
#include "../modules/RTNeural/bench/layer_creator.hpp"

#define RUN_LAYER(Type, size, rand_func)\
    ModelT<float, size, size, Type<float, size, size>> model;\
    rand_func (model.get<0>());\
    duration = run_layer(model);

#define RUN_CONV(size, rand_func)\
    ModelT<float, size, size, Conv1DT<float, size, size, size-1, 1>> model;\
    rand_func (model.get<0>(), size-1);\
    duration = run_layer(model);

#define RUN_ACTIVATION(Type, size)\
    ModelT<float, size, size, Type<float, size>> model;\
    duration = run_layer(model);

using clock_tt = std::chrono::high_resolution_clock;
using second_tt = std::chrono::duration<double>;

std::vector<std::vector<float>> generate_signal(size_t n_samples,
    size_t in_size)
{
    std::vector<std::vector<float>> signal(n_samples);
    for(auto& x : signal)
        x.resize(in_size, 0.0f);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for(size_t i = 0; i < n_samples; ++i)
        for(size_t k = 0; k < in_size; ++k)
            signal[i][k] = distribution(generator);

    return std::move(signal);
}

double rtneural_bench_dynamic(const std::string &layer_type, size_t size, size_t n_samples)
{
    // create layer
    auto layer = create_layer<float>(layer_type, size, size);
    if(layer == nullptr)
        return -1.0;

    // generate audio
    const auto signal = generate_signal(n_samples, size);
    std::vector<float> output(size);

    // run benchmark
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double>;

    auto start = clock_t::now();
    for(size_t i = 0; i < n_samples; ++i)
        layer->forward(signal[i].data(), output.data());
    auto duration = std::chrono::duration_cast<second_t>(clock_t::now() - start).count();
    
    return duration;
}

double rtneural_bench (const std::string &layer_type, size_t size, size_t n_samples) {
    using namespace RTNeural;
    
    auto signal = generate_signal(n_samples, size);
    double duration;

    auto run_layer = [=, &signal] (auto& layer) -> double
    {
        auto start = clock_tt::now();
        for(size_t i = 0; i < n_samples; ++i)
            layer.forward(signal[i].data());
        return std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
    };

    if(layer_type == "dense")
    {
        if(size == 4)
        {
            RUN_LAYER(DenseT, 4, randomise_dense<float>)
        }
        else if(size == 8)
        {
            RUN_LAYER(DenseT, 8, randomise_dense<float>)
        }
        else if(size == 16)
        {
            RUN_LAYER(DenseT, 16, randomise_dense<float>)
        }
        else if(size == 32)
        {
            RUN_LAYER(DenseT, 32, randomise_dense<float>)
        }
        else if(size == 64)
        {
            RUN_LAYER(DenseT, 64, randomise_dense<float>)
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "conv1d")
    {
        if(size == 4)
        {
            RUN_CONV(4, randomise_conv1d<float>)
        }
        else if(size == 8)
        {
            RUN_CONV(8, randomise_conv1d<float>)
        }
        else if(size == 16)
        {
            RUN_CONV(16, randomise_conv1d<float>)
        }
        else if(size == 32)
        {
            RUN_CONV(32, randomise_conv1d<float>)
        }
        else if(size == 64)
        {
            RUN_CONV(64, randomise_conv1d<float>)
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "gru")
    {
        if(size == 4)
        {
            RUN_LAYER(GRULayerT, 4, randomise_gru<float>)
        }
        else if(size == 8)
        {
            RUN_LAYER(GRULayerT, 8, randomise_gru<float>)
        }
        else if(size == 16)
        {
            RUN_LAYER(GRULayerT, 16, randomise_gru<float>)
        }
        else if(size == 32)
        {
            RUN_LAYER(GRULayerT, 32, randomise_gru<float>)
        }
        else if(size == 64)
        {
            RUN_LAYER(GRULayerT, 64, randomise_gru<float>)
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "lstm")
    {
        if(size == 4)
        {
            RUN_LAYER(LSTMLayerT, 4, randomise_lstm<float>)
        }
        else if(size == 8)
        {
            RUN_LAYER(LSTMLayerT, 8, randomise_lstm<float>)
        }
        else if(size == 16)
        {
            RUN_LAYER(LSTMLayerT, 16, randomise_lstm<float>)
        }
        else if(size == 32)
        {
            RUN_LAYER(LSTMLayerT, 32, randomise_lstm<float>)
        }
        else if(size == 64)
        {
            RUN_LAYER(LSTMLayerT, 64, randomise_lstm<float>)
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "tanh")
    {
        if(size == 4)
        {
            RUN_ACTIVATION(TanhActivationT, 4)
        }
        else if(size == 8)
        {
            RUN_ACTIVATION(TanhActivationT, 8)
        }
        else if(size == 16)
        {
            RUN_ACTIVATION(TanhActivationT, 16)
        }
        else if(size == 32)
        {
            RUN_ACTIVATION(TanhActivationT, 32)
        }
        else if(size == 64)
        {
            RUN_ACTIVATION(TanhActivationT, 64)
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "relu")
    {
        if(size == 4)
        {
            RUN_ACTIVATION(ReLuActivationT, 4)
        }
        else if(size == 8)
        {
            RUN_ACTIVATION(ReLuActivationT, 8)
        }
        else if(size == 16)
        {
            RUN_ACTIVATION(ReLuActivationT, 16)
        }
        else if(size == 32)
        {
            RUN_ACTIVATION(ReLuActivationT, 32)
        }
        else if(size == 64)
        {
            RUN_ACTIVATION(ReLuActivationT, 64)
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "sigmoid")
    {
        if(size == 4)
        {
            RUN_ACTIVATION(SigmoidActivationT, 4)
        }
        else if(size == 8)
        {
            RUN_ACTIVATION(SigmoidActivationT, 8)
        }
        else if(size == 16)
        {
            RUN_ACTIVATION(SigmoidActivationT, 16)
        }
        else if(size == 32)
        {
            RUN_ACTIVATION(SigmoidActivationT, 32)
        }
        else if(size == 64)
        {
            RUN_ACTIVATION(SigmoidActivationT, 64)
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }

    return duration;
}

#undef RUN_LAYER
#undef RUN_CONV
#undef RUN_ACTIVATION
