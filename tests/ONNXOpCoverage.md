# Tracking why operators are not covered
[ONNX backend test script](https://github.com/onnx/onnx-caffe2/blob/master/tests/onnx_backend_test.py)
reports the coverage on the operators and attributes. But we have various of reasons for the missing test coverage on operators.
This doc keeps tracking why operators are not covered by the testcases.

| Operator | Test Coverage | PyTorch | Caffe2 |
|---|:--:|:---:|:---:|
|Abs|Yes|OK|OK|
|Add|Yes|OK|OK|
|And||Support int tensor, but no bool tensor|Only support bool tensor|
|ArgMax||||
|ArgMin||||
|AveragePool|Yes|OK|OK|
|BatchNormalization|Yes|OK|OK|
|Cast||||
|Ceil||||
|Clip|Yes|OK|OK|
|Concat|Yes|OK|OK|
|Constant|Yes|OK|OK|
|Conv|Yes|OK|OK|
|ConvTranspose||||
|DepthToSpace||||
|Div|Yes|OK|OK|
|Dropout|Yes|OK|OK|
|Elu|Yes|OK|OK|
|Equal|Only support floating tensor|OK|Only supoprt int tensor|
|Exp|Yes|OK|OK|
|Flatten|Yes|OK|No support for axis|
|Floor|||No support|
|GRU||||
|Gather|Yes|OK|OK|
|Gemm|Yes|OK|OK|
|GlobalAveragePool|Yes|No direct mapping|OK|
|GlobalLpPool||||
|GlobalMaxPool||||
|Greater|||Only support int tensor|
|HardSigmoid|||No support|
|Hardmax|||No support|
|InstanceNormalization|||Only support int tensor|
|LRN|Yes|OK|OK|
|LSTM||||
|LeakyRelu|Yes|OK|OK|
|Less|||Only support int tensor|
|Log|Yes|OK|OK|
|LogSoftmax||PyTorch turns logsoftmax to Log and Softmax|No support for logsoftmax|
|LpNormalization||||
|LpPool||||
|MatMul|Yes|OK|OK|
|Max|Yes|OK|OK|
|MaxPool|Yes|OK|OK|
|MaxRoiPool||||
|Mean||||
|Min||OK|No Support|
|Mul|Yes|OK|OK|
|Neg|Yes|OK|OK|
|Not||||
|Or||||
|PRelu|Yes|OK|OK|
|Pad|Yes|OK|OK|
|Pow||OK|Only accept exponent as argument, not an input|
|RNN||||
|RandomNormal||||
|RandomNormalLike||||
|RandomUniform||||
|RandomUniformLike||||
|Reciprocal||||
|ReduceL1||||
|ReduceL2||||
|ReduceLogSum||||
|ReduceLogSumExp||||
|ReduceMax||||
|ReduceMean||||
|ReduceMin||||
|ReduceProd||||
|ReduceSum||||
|ReduceSumSquare||||
|Relu|Yes|OK|OK|
|Reshape|Yes|OK|OK|
|Selu||||
|Sigmoid|Yes|OK|OK|
|Slice|Yes|OK|OK|
|Softmax|Yes|OK|OK|
|Softplus|Yes|OK|OK|
|Softsign||||
|SpaceToDepth||||
|Split|Yes|OK|OK|
|Sqrt||||
|Squeeze||||
|Sub||||
|Sum|Yes|OK|OK|
|Tanh|Yes|OK|OK|
|Tile||||
|Transpose|Yes|OK|OK|
|Xor||||
|experimental ATen||||
|experimental Affine||||
|experimental ConstantFill||||
|experimental Crop||||
|experimental Embedding||||
|experimental FC||||
|experimental GRUUnit||||
|experimental GivenTensorFill||||
|experimental Identity||||
|experimental ImageScaler||||
|experimental MeanVarianceNormalization||||
|experimental ParametricSoftplus||||
|experimental Scale||||
|experimental ScaledTanh||||
|experimental ThresholdedRelu||||
|experimental Upsample||||
