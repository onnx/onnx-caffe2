from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib

from caffe2.python import workspace
from onnx import helper
from onnx.backend.base import namedtupledict

from onnx_caffe2.workspace import Workspace


def dummy_name(seed=[0]):
    res = 'OC2_DUMMY_{}'.format(seed[0])
    seed[0] += 1
    return res


def make_model(graph, **kwargs):
    kwargs.setdefault('producer_name', 'onnx-caffe2')
    return helper.make_model(graph=graph, **kwargs)


def c2_native_run_op(op_def, inputs):
    with Workspace():
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                workspace.FeedBlob(key, value)
        else:
            assert(len(op_def.input) == len(inputs))
            for key, value in zip(op_def.input, inputs):
                workspace.FeedBlob(key, value)

        workspace.RunOperatorOnce(op_def)

        output_names = op_def.output
        output_values = [workspace.FetchBlob(name) for name in output_names]
        return namedtupledict('Outputs', output_names)(*output_values)


def c2_native_run_net(init_net, predict_net, inputs):
    with Workspace():
        if init_net:
            workspace.RunNetOnce(init_net)

        if isinstance(inputs, dict):
            for key, value in inputs.items():
                workspace.FeedBlob(key, value)
        else:
            uninitialized = [input_name
                             for input_name in predict_net.external_input
                             if not workspace.HasBlob(input_name)]
            assert len(uninitialized) == len(inputs)
            for key, value in zip(uninitialized, inputs):
                workspace.FeedBlob(key, value)

        workspace.RunNetOnce(predict_net)

        output_names = predict_net.external_output
        output_values = [workspace.FetchBlob(name) for name in output_names]
        return namedtupledict('Outputs', output_names)(*output_values)
