#pragma once

#include <onnxruntime_cxx_api.h>

// mostly copied from: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/model-explorer/model-explorer.cpp

[[maybe_unused]] std::string print_shape(const std::vector<std::int64_t> &v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int calculate_product(const std::vector<std::int64_t> &v) {
    int total = 1;
    for (auto &i: v) total *= i;
    return total;
}

template<typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape) {
    Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
    return tensor;
}

double onnx_bench(const std::string &layer_type, size_t size, size_t n_samples) {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "onnx-layer-bench"};
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    const auto model_path =
            std::string{BENCH_ROOT_DIR} + "/bench_models/" + layer_type + "_" + std::to_string(size) + ".onnx";
    Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);

    // print name/shape of inputs
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::int64_t> input_shapes;
//    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
//        std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }
    // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
    for (auto &s: input_shapes) {
        if (s < 0) {
            s = 1;
        }
    }

    // print name/shape of outputs
    std::vector<std::string> output_names;
//    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
//        std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
    }

    // Assume model has 1 input node and 1 output node.
    // assert(input_names.size() == 1 && output_names.size() == 1);

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes;
    auto total_number_elements = calculate_product(input_shape);

    // generate random numbers in the range [0, 255]
    std::vector<float> input_tensor_values(total_number_elements);
    std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));

    // double-check the dimensions of the input tensor
    // assert(input_tensors[0].IsTensor() && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    // std::cout << "\ninput_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

    // pass data through model
    std::vector<const char *> input_names_char(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                   [&](const std::string &str) { return str.c_str(); });

    std::vector<const char *> output_names_char(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                   [&](const std::string &str) { return str.c_str(); });

//    std::cout << "Running model..." << std::endl;
    double duration;
    Ort::RunOptions run_options{nullptr};
    try {
        // initial run (probably allocates memory?)
        auto output_tensors = session.Run(run_options, input_names_char.data(), input_tensors.data(),
                                          input_names_char.size(), output_names_char.data(), output_names_char.size());

        auto start = clock_tt::now();
        for (size_t i = 0; i < n_samples; ++i)
            output_tensors = session.Run(run_options, input_names_char.data(), input_tensors.data(),
                                         input_names_char.size(), output_names_char.data(), output_names_char.size());
        duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();

//        std::cout << "Done!" << std::endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specified in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
    } catch (const Ort::Exception &exception) {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }

    return duration;
}
