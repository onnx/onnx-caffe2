from caffe2.proto import caffe2_pb2
from onnx import onnx_pb2
from numpy import np


np_to_onnx_tensor = {
    np.float32: TensorProto.FLOAT,
    np.uint8: TensorProto.UINT8,
    np.int8: TensorProto.INT8,
    np.uint16: TensorProto.UINT16,
    np.int16: TensorProto.INT16,
    np.int32: TensorProto.INT32,
    np.int64: TensorProto.INT64,
    np.bool: TensorProto.BOOL,
    np.float16: TensorProto.FLOAT16,
}

onnx_tensor_to_np = {
    np.float32: TensorProto.FLOAT,
    np.uint8: TensorProto.UINT8,
    np.int8: TensorProto.INT8,
    np.uint16: TensorProto.UINT16,
    np.int16: TensorProto.INT16,
    np.int32: TensorProto.INT32,
    np.int64: TensorProto.INT64,
    np.bool: TensorProto.BOOL,
    np.float16: TensorProto.FLOAT16,
}

c2_tensor_to_onnx_tensor = {
}

onnx_tensor_to_c2_tensor = {
}
