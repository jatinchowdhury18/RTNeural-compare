#include <RTNeural.h>
#include <chrono>
#include <iostream>
#include "torch_layer.hpp"
#include "rtneural_layer.hpp"

void help()
{
    std::cout << "RTNeural layer comparison benchmarks:" << std::endl;
    std::cout << "Usage: rtneural_compare_bench <layer_type> <length> <size>"
              << std::endl;
    std::cout
        << "    Note that for activation layers the out_size argument is ignored."
        << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc < 4 || argc > 5)
    {
        help();
        return 1;
    }

    std::string layer_type = argv[1];
    if(layer_type == "--help")
    {
        help();
        return 1;
    }

    // parse args
    const auto length_seconds = std::atof(argv[2]);
    const auto size = std::atol(argv[3]);
    std::cout << "Benchmarking " << layer_type << " layer, with input size "
              << size << " and output size " << size
              << ", with signal length " << length_seconds << " seconds"
              << std::endl;

    constexpr double sample_rate = 48000.0;
    const auto n_samples = static_cast<size_t>(sample_rate * length_seconds);

    std::cout << "RTNEURAL..." << std::endl;
    auto duration = rtneural_bench(layer_type, size, n_samples);
    std::cout << "Processed " << length_seconds << " seconds of signal in "
              << duration << " seconds" << std::endl;
    std::cout << length_seconds / duration << "x real-time" << std::endl;

    std::cout << "TORCH..." << std::endl;
    duration = torch_bench(layer_type, size, n_samples);
    std::cout << "Processed " << length_seconds << " seconds of signal in "
              << duration << " seconds" << std::endl;
    std::cout << length_seconds / duration << "x real-time" << std::endl;

    return 0;
}
