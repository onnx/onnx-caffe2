from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import helper


def make_model(graph, **kwargs):
    kwargs.setdefault('producer_name', 'onnx-caffe2')
    return helper.make_model(graph=graph, **kwargs)
