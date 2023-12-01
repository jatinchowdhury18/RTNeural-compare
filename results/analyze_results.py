import numpy as np
import matplotlib.pyplot as plt
import copy

base_results = {
    "sizes": [4, 8, 16, 32, 64],
    "dense": [0, 0, 0, 0, 0],
    "conv1d": [0, 0, 0, 0, 0],
    "lstm": [0, 0, 0, 0, 0],
    "gru": [0, 0, 0, 0, 0],
    "tanh": [0, 0, 0, 0, 0],
    "relu": [0, 0, 0, 0, 0],
    "sigmoid": [0, 0, 0, 0, 0]
}

layer_names = {
    "dense": "Dense",
    "conv1d": "Conv1D",
    "lstm": "LSTM",
    "gru": "GRU",
    "tanh": "Tanh",
    "relu": "ReLU",
    "sigmoid": "Sigmoid"
}

def load_file_all(file):
    torch = copy.deepcopy(base_results)
    onnx = copy.deepcopy(base_results)
    tflite = copy.deepcopy(base_results)
    rt_st = copy.deepcopy(base_results)
    rt_dyn = copy.deepcopy(base_results)

    f = open(file, 'r')
    lines = f.readlines()

    for idx in range(0, len(lines), 16):
        info = lines[idx].split(' ')
        ltype = info[1]
        lsize = int(info[6])
        size_idx = base_results["sizes"].index(lsize)

        rt_st_speed = float(lines[idx + 3].split('x')[0])
        rt_dyn_speed = float(lines[idx + 6].split('x')[0])
        torch_speed = float(lines[idx + 9].split('x')[0])
        onnx_speed = float(lines[idx + 12].split('x')[0])
        tflite_speed = float(lines[idx + 15].split('x')[0])

        onnx[ltype][size_idx] = onnx_speed
        torch[ltype][size_idx] = torch_speed
        rt_st[ltype][size_idx] = rt_st_speed
        rt_dyn[ltype][size_idx] = rt_dyn_speed
        tflite[ltype][size_idx] = tflite_speed

    f.close()
    return torch, onnx, tflite, rt_st, rt_dyn

def load_file_rtneural(file):
    rt_st = copy.deepcopy(base_results)
    rt_dyn = copy.deepcopy(base_results)

    f = open(file, 'r')
    lines = f.readlines()

    for idx in range(0, len(lines), 7):
        info = lines[idx].split(' ')
        ltype = info[1]
        lsize = int(info[6])
        size_idx = base_results["sizes"].index(lsize)

        rt_st_speed = float(lines[idx + 3].split('x')[0])
        rt_dyn_speed = float(lines[idx + 6].split('x')[0])

        rt_st[ltype][size_idx] = rt_st_speed
        rt_dyn[ltype][size_idx] = rt_dyn_speed

    f.close()
    return rt_st, rt_dyn

def make_plot(title, file_name, results):
    legend_labels = ['Torch', 'ONNX', 'TFLITE',
     'RTNeural - xsimd (Static)', 'RTNeural - Eigen (Static)', 'RTNeural - STL (Static)',
     'RTNeural - xsimd', 'RTNeural - Eigen', 'RTNeural - STL']

    if ("GRU" in title) or ("LSTM" in title):
        del legend_labels[2]
        del results[2]

    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for i in range(len(results)):
        ax.semilogy(base_results["sizes"], results[i])
    ax.axhline(y = 1, linestyle='--', color='r')
    plt.xscale('log', base=2)

    plt.title(title)
    plt.xlabel('Layer Size')
    plt.ylabel('Speed (real-time factor)')
    plt.grid(True)
    fig.legend(legend_labels, loc="outside right upper")

    plt.savefig(file_name)


torch_results, onnx_results, tflite_results, rt_stl_st, rt_stl_dyn = load_file_all('results/bench_stl.txt')
rt_eigen_st, rt_eigen_dyn = load_file_rtneural('results/bench_eigen.txt')
rt_xsimd_st, rt_xsimd_dyn = load_file_rtneural('results/bench_xsimd.txt')

for k in base_results.keys():
    if k == "sizes":
        continue

    name = layer_names[k]
    make_plot(f'{name} Layer', f'plots/{k}.png',
        [torch_results[k], onnx_results[k], tflite_results[k], rt_xsimd_st[k], rt_eigen_st[k], rt_stl_st[k], rt_xsimd_dyn[k], rt_eigen_dyn[k], rt_stl_dyn[k]])
    
    # make_plot(f'{name} speed comparison (RTNeural run-time)', f'plots/{k}_dynamic.png',
    #     [torch_results[k], onnx_results[k], rt_xsimd_dyn[k], rt_eigen_dyn[k], rt_stl_dyn[k]])
