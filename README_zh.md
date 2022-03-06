keras2go
====

[English](https://github.com/orestonce/keras2go/blob/master/README.md) | 中文

* keras2go 使用go代码重新了实现的[keras2c](https://github.com/f0uriest/keras2c)的功能
* keras2go 是一个可以把keras网络模型转换成纯go语言实现前向传播的工具

快速开始
====

1. 使用git克隆代码仓库 
2. 安装必要的pip包 ``pip install -r requirements.txt``
3. 运行转换工具，将.h5模型转换成 go代码的实现, 并且进行自动测试
````bash
    cd conv_tool
    python -m keras2go --num_tests 15 --model_path ./model.h5 --function_name Example --package_name example
    go fmt *.go
    go test -v .
````

keras2go 转换命令的使用方法:

````bash
    python -m keras2go [-h] [--num_tests] 10 --model_path ./model.h5 --function_name Example2 --package_name example
    
    arguments:
      -t, --num_tests       生成的测试数据组数量,默认为10
      -m, --model_path      h5模型的文件路径
      -f, --function_name   生成的go语言模型的函数名
      -p, --package_name    生成的go语言模型的包名
      -h, --help            帮助文档      
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

发布协议
====
MIT

相似的项目
====
我发现了Github上的一些相似项目:
  * https://github.com/gosha20777/keras2cpp
  * https://github.com/pplonski/keras2cpp
  * https://github.com/moof2k/kerasify
  * https://github.com/Dobiasd/frugally-deep