#ifndef SPANET_ONNX_GLOBAL_INPUT_HPP
#define SPANET_ONNX_GLOBAL_INPUT_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <highfive/H5Easy.hpp>
#include <ryml.hpp>

#include "base_input.hpp"
#include "utils.hpp"

class GlobalInput : public Input {
private:
    xt::xtensor<bool, 2> mask;
    xt::xtensor<float, 3> features;

    std::vector<int64_t> mask_shape;
    std::vector<int64_t> features_shape;

public:
    GlobalInput(
        const std::string& name,
        const ryml::ConstNodeRef &feature_info,
        const HighFive::File &file,
        const long data_limit
    ) : Input{name, feature_info, file, data_limit} {
        mask = xt::ones<bool>({num_events, num_vectors});
        features = xt::zeros<float>({num_events, num_vectors, num_features});

        for (auto i = 0; i < feature_names.size(); ++i) {
            const auto feature_name = feature_names[i];
            const auto data = H5Easy::load<xt::xtensor<float, 1>>(file, feature_key(feature_name));

            auto current_feature = xt::view(features, xt::all(), 0, i);
            current_feature += data;
        }

        if (data_limit > 0) {
            features = xt::view(features, xt::range(xt::placeholders::_, data_limit));
            mask = xt::view(mask, xt::range(xt::placeholders::_, data_limit));
        }

        for (const auto dim : mask.shape()) {
            mask_shape.push_back(dim);
        }

        for (const auto dim : features.shape()) {
            features_shape.push_back(dim);
        }
    }

    xt::xtensor<bool, 2> &get_mask() override {
        return mask;
    }

    xt::xtensor<float, 3> &get_features() override {
        return features;
    }

    std::vector<int64_t>& get_mask_shape() override {
        return mask_shape;
    }

    std::vector<int64_t>& get_features_shape() override {
        return features_shape;
    }
};

#endif //SPANET_ONNX_GLOBAL_INPUT_HPP
