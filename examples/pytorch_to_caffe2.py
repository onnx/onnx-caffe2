from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from torch.autograd import Variable
from onnx_caffe2.backend import Caffe2Backend
from onnx_caffe2.helper import c2_native_run_net, name_inputs, save_caffe2_net, load_caffe2_net, \
    benchmark_caffe2_model, benchmark_pytorch_model

import io
import logging
import numpy as np
import torch
import onnx


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MyModel(torch.nn.Module):
    '''
        This is simple model for demonstration purpose.
        It requires two 2-D tensors as input,
        and returns the multiply of the two inputs.
    '''
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, m1, m2):
        return torch.mm(m1, m2)


# Create a pytorch model.
log.info("Create a PyTorch model.")
pytorch_model = MyModel()

# Make the inputs in tuple format.
inputs = (Variable(torch.randn(3, 4)), Variable(torch.randn(4, 5)))

# Set the device option.
device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)
#if torch.cuda.is_available():
#    log.info("CUDA is detected, use CUDA.")
#    device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
#    pytorch_model.cuda()
#    for i in inputs:
#        i.cuda()
#else:
#    log.info("No CUDA, use CPU instead.")

# Compute the results using the PyTorch model.
log.info("Run the PyTorch model.")
pytorch_results = pytorch_model(*inputs)

# Export an ONNX model.
log.info("Export an ONNX model from the PyTorch model.")
f = io.BytesIO()
torch.onnx.export(pytorch_model, inputs, f, verbose=True)
onnx_model = onnx.ModelProto.FromString(f.getvalue())

# Check whether the onnx_model is valid or not.
log.info("Check the ONNX model.")
onnx.checker.check_model(onnx_model)

# Convert the ONNX model to a Caffe2 model.
log.info("Convert the model to a Caffe2 model.")
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph)

# Save and load the converted Caffe2 model in the protobuf files.
log.info("Save the Caffe2 models as pb files.")
init_file = "./mymodel_init.pb"
predict_file = "./mymodel_predict.pb"
save_caffe2_net(init_net, init_file, output_txt=False)
save_caffe2_net(predict_net, predict_file, output_txt=True)
log.info("Load the Caffe2 models back.")
init_net = load_caffe2_net(init_file)
predict_net = load_caffe2_net(predict_file)

# Prepare the inputs for Caffe2.
caffe2_inputs = [var.data.numpy() for var in inputs]
init_net.device_option.CopyFrom(device_opts)
predict_net.device_option.CopyFrom(device_opts)

# Compute the results using the Caffe2 model.
log.info("Run the Caffe2 model.")
_, caffe2_results = c2_native_run_net(init_net, predict_net, name_inputs(onnx_model, caffe2_inputs))

# Check the decimal precision of the exported Caffe2.
expected_decimal = 5
for p, c in zip([pytorch_results], caffe2_results):
    if device_opts.device_type == caffe2_pb2.CUDA:
        p.cpu()
    np.testing.assert_almost_equal(p.data.cpu().numpy(), c, decimal=expected_decimal)
log.info("The exported model achieves {}-decimal precision.".format(expected_decimal))

benchmark_pytorch_model(pytorch_model, inputs)
benchmark_caffe2_model(init_net, predict_net)
