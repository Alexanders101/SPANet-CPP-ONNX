#ifndef SPANET_ONNX_BASEINPUT_H
#define SPANET_ONNX_BASEINPUT_H

#include <string>
#include <utility>
#include <xtensor/xarray.hpp>

class Input {
protected:
    size_t num_features;
    size_t num_events;
    size_t num_vectors;

    std::vector<std::string> feature_names;

    const std::string name;
    const ryml::ConstNodeRef feature_info;
    const HighFive::File &file;
    const long data_limit;

    std::string feature_key(const std::string &feature_name) {
        return "INPUTS/" + name + "/" + feature_name;
    }

    std::string mask_name;
    std::string features_name;

public:
    Input(const std::string& name, const ryml::ConstNodeRef &feature_info, const HighFive::File &file, const long data_limit)
            : name{name},
              feature_info{feature_info},
              file{file},mask_name{name + "_mask"},
              features_name{name + "_data"},
              data_limit{data_limit}
    {
        for (const auto feature: feature_info) {
            const auto feature_name = to_string(feature.key());
            if (feature_name != "MASK") {
                feature_names.push_back(feature_name);
            }
        }

        const auto first_dataset = file.getDataSet(feature_key(feature_names[0]));
        const auto dimensions = first_dataset.getDimensions();

        num_features = feature_names.size();
        num_events = dimensions[0];

        if (dimensions.size() > 1)
            num_vectors = dimensions[1];
        else
            num_vectors = 1;
    }

    virtual xt::xtensor<bool, 2>& get_mask() = 0;
    virtual xt::xtensor<float, 3>& get_features() = 0;

    virtual std::vector<int64_t>& get_mask_shape() = 0;
    virtual std::vector<int64_t>& get_features_shape() = 0;

    virtual std::string& get_mask_name() {
        return mask_name;
    }

    virtual std::string& get_features_name() {
        return features_name;
    }

    virtual const std::string &get_name() const {
        return name;
    }
};


#endif //SPANET_ONNX_BASEINPUT_H
