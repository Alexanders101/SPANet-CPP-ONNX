#ifndef SPANET_ONNX_UTILS_H
#define SPANET_ONNX_UTILS_H

#include <ryml.hpp>
#include <string>
#include <onnxruntime_cxx_api.h>

std::string read_file(const std::string& filepath);
std::string to_string(const ryml::csubstr& substring);
std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type);

#endif //SPANET_ONNX_UTILS_H
