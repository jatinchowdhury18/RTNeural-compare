#!/bin/bash

set -e

build_bench()
{
    rm -Rf build
    sed -i "8s/.*/set($1 ON CACHE BOOL \"Use RTNeural with this backend\" FORCE)/" CMakeLists.txt
    cmake -Bbuild -DCMAKE_PREFIX_PATH=~/Documents/RTNeural_Compare/modules/libtorch
    cmake --build build --parallel --config Release
}

run_bench()
{
    rm -f $1
    touch $1

    bench=./build/rtneural_compare_bench
    length_seconds=10
    
    layers=("dense" "conv1d" "gru" "lstm" "tanh" "relu" "sigmoid")
    sizes=(4 8 16 32 64)

    for layer in ${layers[@]}; do
        for size in ${sizes[@]}; do
            $bench $layer $length_seconds $size | tee -a $1
        done
    done 
}

build_bench "RTNEURAL_STL"
run_bench "results/bench_stl.txt"

build_bench "RTNEURAL_EIGEN"
run_bench "results/bench_eigen.txt"

build_bench "RTNEURAL_XSIMD"
run_bench "results/bench_xsimd.txt"
