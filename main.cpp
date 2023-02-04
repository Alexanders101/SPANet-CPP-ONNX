#include <iostream>
#include <fstream>

#include <string>

#include <highfive/H5File.hpp>

#include <ryml.hpp>
#include <ryml_std.hpp>
#include <c4/format.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

#include <onnxruntime_cxx_api.h>

#include "utils.hpp"
#include "base_input.hpp"
#include "sequential_input.hpp"
#include "global_input.hpp"

std::vector<std::unique_ptr<Input>> load_data(const c4::yml::Tree &event_info, const H5Easy::File &dataset, const long data_limit = -1) {
    std::vector<std::unique_ptr<Input>> inputs;

    for (const auto input : event_info["INPUTS"]["SEQUENTIAL"]) {
        const auto name = to_string(input.key());
        inputs.emplace_back(new SequentialInput(name, input, dataset, data_limit));
    }

    for (const auto input : event_info["INPUTS"]["GLOBAL"]) {
        const auto name = to_string(input.key());
        inputs.emplace_back(new GlobalInput(name, input, dataset, data_limit));
    }

    return inputs;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Usage: <model_file> <event_file> <input_file> <output_file> [cpu|gpu] [data_limit]" << std::endl;
        return 0;
    }

    std::string model_file(argv[1]);
    std::string event_file(argv[2]);
    std::string input_file(argv[3]);
    std::string output_file(argv[4]);

    std::string device = argc >= 6 ? argv[5] : "cpu";
    auto data_limit = argc >= 7 ? std::stol(argv[6]) : -1;

    const auto event_info = ryml::parse_in_arena(ryml::to_csubstr(read_file(event_file)));
    const auto dataset = H5Easy::File(input_file, HighFive::File::ReadOnly);
    const auto inputs = load_data(event_info, dataset, data_limit);

    for (const auto& input : inputs) {
        std::cout << "===================================================================================" << std::endl;
        std::cout << input->get_name() << ": " << xt::adapt(input->get_features().shape()) << std::endl;
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
        std::cout << input->get_features() << std::endl;
        std::cout << "===================================================================================" << std::endl;
        std::cout << std::endl;
    }

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "SPANet");
    Ort::SessionOptions sessionOptions;

    if (device == "gpu") {
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, model_file.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    std::cout << std::endl;
    std::cout << "===================================================================================" << std::endl;
    std::cout << "Number of Input Nodes: " << num_input_nodes << std::endl;
    for (auto i = 0; i < num_input_nodes; ++i) {
        const auto input_names = session.GetInputNameAllocated(i, allocator);
        const auto input_type_info = session.GetInputTypeInfo(i);
        const auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
        std::cout << "Input Index: " << i << std::endl;
        std::cout << "Input Name: " << std::string(input_names.get()) << std::endl;
        std::cout << "Input Type: " << input_tensor_info.GetElementType() << std::endl;
        std::cout << "Input Shape: " << xt::adapt(input_tensor_info.GetShape()) << std::endl;
    }
    std::cout << "===================================================================================" << std::endl;
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "===================================================================================" << std::endl;
    std::cout << "Number of Output Nodes: " << num_output_nodes << std::endl;
    for (auto i = 0; i < num_output_nodes; ++i) {
        const auto output_name = session.GetOutputNameAllocated(i, allocator);
        const auto output_type_info = session.GetOutputTypeInfo(i);
        const auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
        std::cout << "Output Index: " << i << std::endl;
        std::cout << "Output Name: " << std::string(output_name.get()) << std::endl;
        std::cout << "Output Type: " << output_tensor_info.GetElementType() << std::endl;
        std::cout << "Output Shape: " << xt::adapt(output_tensor_info.GetShape()) << std::endl;
    }
    std::cout << "===================================================================================" << std::endl;
    std::cout << std::endl;

    std::vector<Ort::AllocatedStringPtr> raw_output_names;
    std::vector<const char*> onnx_input_names;
    std::vector<const char*> onnx_output_names;
    std::vector<Ort::Value> onnx_input_tensors;

    Ort::MemoryInfo cpu_memory = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    for (auto& input : inputs) {
        onnx_input_names.push_back(input->get_features_name().c_str());
        onnx_input_tensors.push_back(Ort::Value::CreateTensor<float>(
            cpu_memory,
            input->get_features().data(),
            input->get_features().size(),
            input->get_features_shape().data(),
            input->get_features_shape().size()
        ));

        onnx_input_names.push_back(input->get_mask_name().c_str());
        onnx_input_tensors.push_back(Ort::Value::CreateTensor<bool>(
            cpu_memory,
            input->get_mask().data(),
            input->get_mask().size(),
            input->get_mask_shape().data(),
            input->get_mask_shape().size()
        ));

    }

    for (auto i = 0; i < num_output_nodes; ++i) {
        raw_output_names.emplace_back(session.GetOutputNameAllocated(i, allocator));
        onnx_output_names.push_back(raw_output_names[i].get());
    }

    auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            onnx_input_names.data(),
            onnx_input_tensors.data(),
            onnx_input_tensors.size(),
            onnx_output_names.data(),
            onnx_output_names.size()
    );

    for (auto i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];

        const auto data = xt::adapt(
            output.GetTensorData<float>(),
            output.GetTensorTypeAndShapeInfo().GetShape()
        );

        std::cout << "===================================================================================" << std::endl;
        std::cout << std::string(onnx_output_names[i]) << ": " << xt::adapt(output.GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
        std::cout << data << std::endl;
        std::cout << "===================================================================================" << std::endl;
        std::cout << std::endl;

    }

    return 0;
}
