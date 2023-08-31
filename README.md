# RTNeural Comparisons

This repository contains an executable for comparing the performance
of C++ neural network inferencing engines. Currently, there are four
inferencing engines being compared:

- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) (compile-time API)
- RTNeural (run-time API)
- [libtorch](https://pytorch.org/cppdocs/)
- [onnxruntime](https://github.com/microsoft/onnxruntime)

## Results
All benchmarks results were obtained on a 2018 Mac Mini with an Intel(R) Core(TM) i7-8700B CPU @ 3.20GHz.
The "Real-Time Factor" measurement assumes a sample rate of 48 kHz.

### Dense:

![](./plots/dense_static.png)
![](./plots/dense_dynamic.png)

### Conv1D:

![](./plots/conv1d_static.png)
![](./plots/conv1d_dynamic.png)

### GRU:

![](./plots/gru_static.png)
![](./plots/gru_dynamic.png)

### LSTM:

![](./plots/lstm_static.png)
![](./plots/lstm_dynamic.png)

### Activations:

Tanh:

![](./plots/tanh_static.png)
![](./plots/tanh_dynamic.png)

ReLU:

![](./plots/relu_static.png)
![](./plots/relu_dynamic.png)

Sigmoid:

![](./plots/sigmoid_static.png)
![](./plots/sigmoid_dynamic.png)
