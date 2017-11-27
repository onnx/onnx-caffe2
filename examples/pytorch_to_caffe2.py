from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from torch.autograd import Variable
from onnx_caffe2.backend import Caffe2Backend
from onnx_caffe2.helper import c2_native_run_net, save_caffe2_net, load_caffe2_net, \
    benchmark_caffe2_model, benchmark_pytorch_model

import io
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Create a pytorch model.
log.info("Create a PyTorch model.")
pytorch_model = MNIST()
pytorch_model.train(False)

# Make the inputs in tuple format.
inputs = (Variable(torch.randn(3, 1, 28, 28), requires_grad=True), )

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
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph, device="CPU")

# Caffe2 model takes a numpy array list as input.
caffe2_inputs = [var.data.numpy() for var in inputs]

# Save and load the converted Caffe2 model in the protobuf files.
log.info("Save the Caffe2 models as pb files.")
init_file = "./mymodel_init.pb"
predict_file = "./mymodel_predict.pb"
save_caffe2_net(init_net, init_file, output_txt=False)
save_caffe2_net(predict_net, predict_file, output_txt=True)
log.info("Load the Caffe2 models back.")
init_net = load_caffe2_net(init_file)
predict_net = load_caffe2_net(predict_file)

# Compute the results using the PyTorch model.
log.info("Run the PyTorch model.")
pytorch_results = pytorch_model(*inputs)

# Compute the results using the Caffe2 model.
log.info("Run the Caffe2 model.")
_, caffe2_results = c2_native_run_net(init_net, predict_net, caffe2_inputs)

# Check the decimal precision of the exported Caffe2.
expected_decimal = 5
for p, c in zip([pytorch_results], caffe2_results):
    np.testing.assert_almost_equal(p.data.cpu().numpy(), c, decimal=expected_decimal)
log.info("The exported model achieves {}-decimal precision.".format(expected_decimal))

pytorch_time = benchmark_pytorch_model(pytorch_model, inputs)
caffe2_time = benchmark_caffe2_model(init_net, predict_net)

print("PyTorch model's execution time is {} milliseconds/ iteration, {} iterations per second.".format(
    pytorch_time, 1000 / pytorch_time))
print("Caffe2 model's execution time is {} milliseconds / iteration, {} iterations per second".format(
    caffe2_time, 1000 / caffe2_time))
