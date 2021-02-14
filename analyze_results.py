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

def load_file(file):
    torch = copy.deepcopy(base_results)
    rt = copy.deepcopy(base_results)

    f = open(file, 'r')
    lines = f.readlines()

    for idx in range(0, len(lines), 7):
        info = lines[idx].split(' ')
        ltype = info[1]
        lsize = int(info[6])
        size_idx = base_results["sizes"].index(lsize)

        rt_speed = float(lines[idx + 3].split('x')[0])
        torch_speed = float(lines[idx + 6].split('x')[0])

        torch[ltype][size_idx] = torch_speed
        rt[ltype][size_idx] = rt_speed

    f.close()
    return torch, rt

torch_results, rt_stl_results = load_file('results/bench_stl.txt')
_, rt_eigen_results = load_file('results/bench_eigen.txt')
_, rt_xsimd_results = load_file('results/bench_xsimd.txt')

for k in base_results.keys():
    if k == "sizes":
        continue

    rt_best = [max(*l) for l in zip(rt_stl_results[k], rt_eigen_results[k], rt_xsimd_results[k])]

    plt.figure()
    plt.semilogy(base_results["sizes"], torch_results[k])
    plt.semilogy(base_results["sizes"], rt_best)
    plt.axhline(y = 1, linestyle='--', color='r')

    plt.title(f'{k} speed comparison')
    plt.xlabel('Size')
    plt.ylabel('Speed (real-time factor)')
    plt.legend(['Torch', 'RTNeural'])

    plt.savefig(f'plots/{k}.png')
