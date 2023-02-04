# SPANet C++ ONNX Example

A simple example of how to load a model exported by [SPANet](https://github.com/Alexanders101/SPANet) and use it inside of a C++ application.

## Building Instructions

This repo is more or less self-sustaining, with code dependencies included as submodules. 
First, grab the latest ONNX runtime from [https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases).
Install it to a known location and set `export ONNXRUNTIME_ROOTDIR="..."`

Then run the following commands
```bash
git clone --recursive https://github.com/Alexanders101/SPANet-CPP-ONNX
cd SPANet-CPP-ONNX
mkdir build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_USE_CUDA=ON \
  -DONNXRUNTIME_ROOTDIR=${ONNXRUNTIME_ROOTDIR} \
  -S $(pwd) \
  -B $(pwd)/build
  
cmake --build $(pwd)/build --target spanet_onnx -- -j 8
./build/spanet_onnx example/tth.onnx example/semi_leptonic_ttH.yaml example/example.h5 example/output.h5
```

## Acknowledgement
Big thanks to the following repo for providing some examples on using the ONNX C++ API: [https://github.com/leimao/ONNX-Runtime-Inference](https://github.com/leimao/ONNX-Runtime-Inference) 
