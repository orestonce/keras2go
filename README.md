keras2go
====

English | [中文](https://github.com/orestonce/keras2go/blob/master/README_zh.md)

* keras2go uses go code to re-implement the functionality of [keras2c](https://github.com/f0uriest/keras2c)
* keras2go is a library for deploying keras neural networks in pure go, using only standard libraries. It is designed to be as simple as possible for real time applications.

Quickstart
====

After cloning the repo, install the necessary packages with ``pip install -r requirements.txt``.
1. Clone the repo
2. install the necessary packages ``pip install -r requirements.txt``
3. Run the conversion tool to convert the .h5 model to the implementation of the go code, then run go test
````bash
    cd conv_tool
    python -m keras2go --num_tests 15 --model_path ./model.h5 --function_name Example --package_name example
    go fmt *.go
    go test -v .
````

keras2go can be used from the command line:


````bash
    python -m keras2go [-h] [--num_tests] 10 --model_path ./model.h5 --function_name Example2 --package_name example

    A library for converting the forward pass (inference) part of a keras model to a go function
    arguments:
      -t, --num_tests       Number of tests to generate. Default is 10
      -m, --model_path      File path to saved keras .h5 model file
      -f, --function_name   What to name the resulting go function
      -p, --package_name    What to name the resulting go package
      -h, --help            show this help message and exit      
````

Supported Layers
====
  - Core Layers: Dense, Activation, Dropout, Flatten, Input, Reshape, Permute, RepeatVector,  ActivityRegularization, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D
  - Convolution Layers: Conv1D, Conv2D, Conv3D, Cropping1D, Cropping2D, Cropping3D, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
  - Pooling Layers: MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling3D,GlobalAveragePooling3D
  - Recurrent Layers: SimpleRNN, GRU, LSTM, SimpleRNNCell, GRUCell, LSTMCell
  - Embedding Layers: Embedding
  - Merge Layers: Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot
  - Advanced Activation Layers: LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ReLU
  - Normalization Layers: BatchNormalization
  - Noise Layers: GaussianNoise, GaussianDropout, AlphaDropout
  - Layer Wrappers: TimeDistributed, Bidirectional
  
ToDo
====
  - test code
  - Core Layers: Lambda, Masking
  - Convolution Layers: SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, Conv3DTranspose
  - Pooling Layers: MaxPooling3D, AveragePooling3D
  - Locally Connected Layers: LocallyConnected1D, LocallyConnected2D
  - Recurrent Layers: ConvLSTM2D, ConvLSTM2DCell
  - Merge Layers: Broadcasting merge between different sizes
  - Misc: models made from submodels

License
====
MIT


Similar projects
====
I found another similar projects on Github:
  * https://github.com/gosha20777/keras2cpp
  * https://github.com/pplonski/keras2cpp
  * https://github.com/moof2k/kerasify
  * https://github.com/Dobiasd/frugally-deep
