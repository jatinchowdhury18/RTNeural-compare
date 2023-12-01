#pragma once

#include <string>
#include <vector>
#include <tensorflow/lite/c_api.h>

// mostly copied from: https://github.com/Torsion-Audio/nn-inference-template/blob/main/minimal-inference/tensorflow-lite/minimal-tflite.cpp

double tflite_bench(const std::string &layer_type, size_t size, size_t n_samples) {
    if ((layer_type.find ("gru") != std::string::npos) || (layer_type.find ("lstm") != std::string::npos)) {
//        std::cout << "TFLITE: skipping recurrent layers!" << std::endl;
        return -1.0;
    }

    // Load model
    const auto model_path =
            std::string{BENCH_ROOT_DIR} + "/bench_models/" + layer_type + "_" + std::to_string(size) + ".tflite";
    TfLiteModel *model = TfLiteModelCreateFromFile(model_path.c_str());

    // Create the interpreter
    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

    // Allocate memory for all tensors
    TfLiteInterpreterAllocateTensors(interpreter);

    // Get input tensor
    TfLiteTensor *inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    // Fill input tensor with data
    std::vector<float> input_data(size);
    std::vector<float> output_data(size);
    std::generate(input_data.begin(), input_data.end(), [&] { return rand() % 255; });

    auto start = clock_tt::now();
    for (size_t n = 0; n < n_samples; ++n)
    {
        TfLiteTensorCopyFromBuffer(inputTensor, input_data.data(), size * sizeof(float));

        // Execute inference.
        TfLiteInterpreterInvoke(interpreter);

        // Get output tensor
        const TfLiteTensor *outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

        // Extract the output tensor data
        TfLiteTensorCopyToBuffer(outputTensor, output_data.data(), size * sizeof(float));
    }

    auto duration = std::chrono::duration_cast<second_tt>(clock_tt::now() - start).count();

    // Dispose of the model and interpreter objects.
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return duration;
}
