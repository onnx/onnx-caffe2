from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from onnx import helper
from onnx.backend.base import namedtupledict

from onnx_caffe2.workspace import Workspace

import io
import logging
import time


log = logging.getLogger(__name__)


class _DummyNameFactory(object):
    used_names = set()
    counter = 0

    @classmethod
    def dummy_name(cls, used_names=None):
        if used_names is not None:
            cls.used_names.clear()
            cls.used_names.update(used_names)
            cls.counter = 0
            return None
        else:
            while True:
                name = 'OC2_DUMMY_{}'.format(cls.counter)
                cls.counter += 1
                if name not in cls.used_names:
                    cls.used_names.add(name)
                    return name

dummy_name = _DummyNameFactory.dummy_name


def make_model(graph, **kwargs):
    kwargs.setdefault('producer_name', 'onnx-caffe2')
    return helper.make_model(graph=graph, **kwargs)


def c2_native_run_op(op_def, inputs):
    ws = Workspace()
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            ws.FeedBlob(key, value, op_def.device_option)
    else:
        assert(len(op_def.input) == len(inputs))
        for key, value in zip(op_def.input, inputs):
            ws.FeedBlob(key, value, op_def.device_option)

    ws.RunOperatorOnce(op_def)

    output_names = op_def.output
    output_values = [ws.FetchBlob(name) for name in output_names]
    return ws, namedtupledict('Outputs', output_names)(*output_values)


def c2_native_run_net(init_net, predict_net, inputs):
    ws = Workspace()
    if init_net:
        ws.RunNetOnce(init_net)

    if isinstance(inputs, dict):
        for key, value in inputs.items():
            ws.FeedBlob(key, value, predict_net.device_option)
    else:
        uninitialized = [input_name
                         for input_name in predict_net.external_input
                         if not ws.HasBlob(input_name)]
        assert len(uninitialized) == len(inputs)
        for key, value in zip(uninitialized, inputs):
            ws.FeedBlob(key, value, predict_net.device_option)

    ws.RunNetOnce(predict_net)

    output_names = predict_net.external_output
    output_values = [ws.FetchBlob(name) for name in output_names]
    return ws, namedtupledict('Outputs', output_names)(*output_values)


def name_inputs(onnx_model, inputs):
    '''
        Pair the numpy arrays in inputs with their names in onnx_model.
        The names are usually needed for Caffe2 blobs initialization.
    '''
    return {onnx_model.graph.input[i].name: inputs[i] for i in range(len(inputs))}


def benchmark_caffe2_model(init_net, predict_net, warmup_iters=3, main_iters=10, layer_details=True):
    ws = Workspace()
    if init_net:
        ws.RunNetOnce(init_net)
    ws.CreateNet(predict_net)
    ws.BenchmarkNet(predict_net.name, warmup_iters, main_iters, layer_details)


def load_caffe2_net(file):
    net = caffe2_pb2.NetDef()
    with open(file, "rb") as f:
        net.ParseFromString(f.read())
    return net


def save_caffe2_net(net, file, output_txt=False):
    with open(file, "wb") as f:
        f.write(net.SerializeToString())
    if output_txt:
        with open(file + "txt", "w") as f:
            f.write(str(net))


def benchmark_pytorch_model(model, inputs, training=False, warmup_iters=3,
                            main_iters=10, verbose=False):
    for _i in range(warmup_iters):
        ts = time.time()
        model(*inputs)
        te = time.time()
    total_pytorch_time = 0.0
    for _i in range(main_iters):
        ts = time.time()
        model(*inputs)
        te = time.time()
        total_pytorch_time = te - ts + total_pytorch_time
    log.info("The PyTorch model execution time per iter is {} milliseconds, "
             "{} iters per second.".format(total_pytorch_time / main_iters * 1000,
                                           main_iters / total_pytorch_time))
