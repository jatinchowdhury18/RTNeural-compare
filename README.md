# RTNeural Comparisons

This repository contains an executable for comparing the performance
of C++ neural network inferencing engines. Currently, there are three
inferencing engines being compared:

- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) (compile-time API)
- RTNeural (run-time API)
- [libtorch](https://pytorch.org/cppdocs/)

## Results
All benchmarks results were obtained on a Macbook with an Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz.

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
