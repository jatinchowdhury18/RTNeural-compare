#pragma once

#include <torch/torch.h>
#include <chrono>
#include <iostream>

using clock_tt = std::chrono::high_resolution_clock;
using second_tt = std::chrono::duration<double>;

double torch_bench (const std::string &layer_type, size_t size, size_t n_samples) {
  auto signal = torch::rand({1,1,(int)size});
  double duration;

  if (layer_type == "dense") {
    torch::nn::Linear layer {size, size};
    torch::nn::init::uniform_(layer->weight, -1.0, 1.0);
    torch::nn::init::uniform_(layer->bias, -1.0, 1.0);

    auto start = clock_tt::now();
    for(size_t i = 0; i < n_samples; ++i)
      layer->forward(signal);
    duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
  }

  if (layer_type == "conv1d") {
    signal = torch::rand({1,(int)size,(int)size});
    auto layer = torch::nn::Conv1d(
                torch::nn::Conv1dOptions(size,size,size - 1)
                .stride(1)
                .dilation(1)
                .bias(true));
    torch::nn::init::uniform_(layer->weight, -1.0, 1.0);
    torch::nn::init::uniform_(layer->bias, -1.0, 1.0);
    
    auto start = clock_tt::now();
    for(size_t i = 0; i < n_samples; ++i)
      layer->forward(signal);
    duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
  }

  if (layer_type == "gru") {
    torch::nn::GRU layer {size, size};
    for (auto& w : layer->all_weights())
      torch::nn::init::uniform_(w, -1.0, 1.0);

    auto start = clock_tt::now();
    for(size_t i = 0; i < n_samples; ++i)
      layer->forward(signal);
    duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
  }

  if (layer_type == "lstm") {
    torch::nn::LSTM layer {size, size};
    for (auto& w : layer->all_weights())
      torch::nn::init::uniform_(w, -1.0, 1.0);

    auto start = clock_tt::now();
    for(size_t i = 0; i < n_samples; ++i)
      layer->forward(signal);
    duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
  }

  if (layer_type == "tanh") {
    auto start = clock_tt::now();
    for(size_t i = 0; i < n_samples; ++i)
      torch::tanh(signal);
    duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
  }

  if (layer_type == "relu") {
    auto start = clock_tt::now();
    for(size_t i = 0; i < n_samples; ++i)
      torch::relu(signal);
    duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
  }

  if (layer_type == "sigmoid") {
    auto start = clock_tt::now();
    for(size_t i = 0; i < n_samples; ++i)
      torch::sigmoid(signal);
    duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();
  }

  return duration;
}
