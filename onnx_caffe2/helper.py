from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib

from onnx import helper
from onnx.backend.base import namedtupledict

from onnx_caffe2.workspace import Workspace


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
            ws.FeedBlob(key, value)
    else:
        assert(len(op_def.input) == len(inputs))
        for key, value in zip(op_def.input, inputs):
            ws.FeedBlob(key, value)

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
            ws.FeedBlob(key, value)
    else:
        uninitialized = [input_name
                         for input_name in predict_net.external_input
                         if not ws.HasBlob(input_name)]
        assert len(uninitialized) == len(inputs)
        for key, value in zip(uninitialized, inputs):
            ws.FeedBlob(key, value)

    ws.RunNetOnce(predict_net)

    output_names = predict_net.external_output
    output_values = [ws.FetchBlob(name) for name in output_names]
    return ws, namedtupledict('Outputs', output_names)(*output_values)
