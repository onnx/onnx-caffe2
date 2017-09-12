from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

caffe2 = pytest.importorskip("caffe2")

import os
import sys
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import brew
from caffe2.python.model_helper import ModelHelper

import onnx
from onnx import helper
from onnx import onnx_pb2
import onnx_caffe2.frontend as c2_onnx
import onnx_caffe2.backend as c2

import numpy as np
import google.protobuf.text_format
import unittest
from caffe2.python.models.download import downloadFromURLToFile, getURLFromName, deleteDirectory


class NodeSpec(object):
    """
    Describes a onnx.NodeProto, but without inputs/outputs
    (which will be inferred).
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


N = NodeSpec


def create_input(call_args):
    """
    Creates a tuple of Numpy ndarray from a template 'call_args'; every embedded
    tuple in call_args is converted into a random ndarray with the specified
    dimensions.
    """
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        if isinstance(arg, tuple):
            return np.random.randn(*arg).astype(np.float32)
        else:
            return arg

    return tuple(map_arg(arg) for arg in call_args)


class Caffe2NumpySpec(object):
    """
    A test for a single ONNX node against a Numpy reference implementation.
    """
    def __init__(self, name, node, np_impl, inputs):
        """
        Arguments:
            name (string): Eventual name of the test.  Must be prefixed
                with 'test_'.
            node (NodeSpec): The ONNX node's name and attributes to be tested;
                inputs and outputs will be inferred from other arguments of this
                spec.
            np_impl (lambda): A function from any number of Numpy ndarrays,
                to a single ndarray or tuple of ndarrays.
            inputs (tuple of ndarrays or size tuples): A specification of
                the input to the operator.
        """
        # We don't prepend the 'test_' prefix to improve greppability
        if not name.startswith("test_"):
            raise ValueError("Test name must start with test_")
        self.name = name
        self.node = node
        self.np_impl = np_impl
        self.inputs = inputs

    def run(self, test_self):
        # TODO: In some cases we should generate multiple random inputs
        # and test (ala Hypothesis)
        args = create_input(self.inputs)
        np_results = self.np_impl(*args)
        if not isinstance(np_results, tuple):
            np_results = (np_results, )
        input_names = ["input_{}".format(i) for i in range(len(args))]
        output_names = ["output_{}".format(i) for i in range(len(np_results))]
        node_def = helper.make_node(self.node.name, input_names, output_names, **self.node.kwargs)
        caffe2_results = c2.run_node(node_def, args)
        test_self.assertEqual(len(np_results), len(caffe2_results))
        for i in range(len(output_names)):
            np.testing.assert_almost_equal(np_results[i], caffe2_results[i])

CN = Caffe2NumpySpec


L = 20
M = 10
S = 5
const2_np = np.random.randn(S, S)
const2_onnx = onnx.helper.make_tensor("const2", onnx.TensorProto.FLOAT, (S, S), const2_np.flatten().astype(float))

# TODO: These Numpy specs will be generally useful to backend implementations,
# so they should get moved out of here at some point
#
# NB: There is some degree of duplication between this and
# 'caffe2/caffe2/python/operator_test', but these are against ONNX nodes,
# not Caffe2 nodes.
#
# TODO: Some of these tests will be done most conveniently by running a Caffe2
# operator directly (i.e., something like a before-after translation). If you
# need this, extend the test runner to handle these conveniently; possibly
# use Caffe2's immediate mode
node_tests = [
  CN("test_abs", N("Abs"), np.abs, inputs=((S, S, S),)),
  CN("test_add", N("Add"), np.add, inputs=((S, S, S), (S, S, S))),
  CN("test_add_bcast", N("Add", broadcast=1), np.add, inputs=((S, M), (S))),
  CN("test_constant", N("Constant", value=const2_onnx), lambda: const2_np, inputs=()),
  # TODO: Are we actually supporting other dot modes?  In that case, some fancy
  # footwork is necessary...
  CN("test_dot", N("Dot"), np.dot, inputs=((S, M), (M, L))),
  CN("test_relu", N("Relu"), lambda x: np.clip(x, 0, np.inf), inputs=((S, S, S),)),
  # TODO: Add all the other operators
  ]


class TestCaffe2Reference(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=0)

    def test_relu_node(self):
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"])
        X = np.random.randn(3, 2).astype(np.float32)
        # Testing with a list
        output = c2.run_node(
            node_def, [X])
        Y_ref = np.clip(X, 0, np.inf)
        np.testing.assert_almost_equal(output["Y"], Y_ref)
        # Testing with a dictionary
        output = c2.run_node(
            node_def, {"X": X})
        Y_ref = np.clip(X, 0, np.inf)
        np.testing.assert_almost_equal(output["Y"], Y_ref)

    def test_relu_node_inplace(self):
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], consumed_inputs=[1])
        X = np.random.randn(3, 2).astype(np.float32)
        output = c2.run_node(
            node_def, {"X": X})
        graph_def = helper.make_graph(
            [node_def],
            name="test",
            inputs=["X"],
            outputs=["X", "Y"])
        Y_ref = np.clip(X, 0, np.inf)
        c2_rep = c2.prepare(graph_def)
        output = c2_rep.run({"X": X})
        # With the inplace change from Zach, there shouldn't be Y
        # np.testing.assert_almost_equal(output["Y"], Y_ref)

        # ensure  we wrote over X
        np.testing.assert_almost_equal(output["X"], Y_ref)

    def test_relu_graph(self):
        inputs = ['X']
        outputs = ['Y']
        graph_def = helper.make_graph(
            [helper.make_node("Relu", inputs, outputs)],
            name="test",
            inputs=inputs,
            outputs=outputs)
        X = np.random.randn(3, 2).astype(np.float32)
        Y_ref = np.clip(X, 0, np.inf)
        # Testing with a list
        c2_rep = c2.prepare(graph_def)
        output = c2_rep.run({"X": X})
        np.testing.assert_almost_equal(output["Y"], Y_ref)


    def test_initializer(self):
        X = np.array([[1, 2], [3, 4]]).astype(np.float32)
        Y = np.array([[1, 2], [3, 4]]).astype(np.float32)
        weight = np.array([[1, 0], [0, 1]])
        graph_def = helper.make_graph(
            [helper.make_node("Add", ["X", "Y"], ["Z0"]),
             helper.make_node("Cast", ["Z0"], ["Z"], to="float"),
             helper.make_node("Mul", ["Z", "weight"], ["W"]),
             helper.make_node("Tanh", ["W"], ["W"]),
             helper.make_node("Sigmoid", ["W"], ["W"]),
             helper.make_node("Scale", ["W"], ["W"], scale=-1.0)],
            name="test_initializer",
            inputs=["X", "Y", "weight"],
            outputs=["W"],
            initializer=[helper.make_tensor("weight", onnx_pb2.TensorProto.FLOAT, [2, 2], weight.flatten().astype(float))]
        )
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        W_ref = -sigmoid(np.tanh((X + Y) * weight))
        c2_rep = c2.prepare(graph_def)
        output = c2_rep.run({"X": X, "Y": Y})
        np.testing.assert_almost_equal(output["W"], W_ref)

    def model_dir(self, model):
        caffe2_home = os.path.expanduser(os.getenv('CAFFE2_HOME', '~/.caffe2'))
        models_dir = os.getenv('CAFFE2_MODELS', os.path.join(caffe2_home, 'models'))
        return os.path.join(models_dir, model)

    def _test_net(self, net_name, input_blob_dims=[1, 3, 224, 224], decimal=7):
        model_dir = self.model_dir(net_name)
        # predict net is stored as a protobuf text
        c2_predict_pb = os.path.join(model_dir, 'predict_net.pbtxt')
        c2_predict_net = caffe2_pb2.NetDef()
        with open(c2_predict_pb, 'r') as f:
            google.protobuf.text_format.Merge(f.read(), c2_predict_net)
        c2_predict_net.name = net_name

        # init net(weights) is stored as a protobuf binary
        c2_init_pb = os.path.join(model_dir, 'init_net.pb')
        c2_init_net = caffe2_pb2.NetDef()
        with open(c2_init_pb, 'rb') as f:
            c2_init_net.ParseFromString(f.read())
        c2_init_net.name = net_name + '_init'

        n, c, h, w = input_blob_dims
        data = np.random.randn(n, c, h, w).astype(np.float32)
        inputs = [data]
        c2_ref = c2_onnx.caffe2_net_reference(c2_init_net, c2_predict_net, inputs)

        init_graph = c2_onnx.caffe2_net_to_onnx_graph(c2_init_net)
        predict_graph = c2_onnx.caffe2_net_to_onnx_graph(c2_predict_net)
        c2_ir = c2.prepare(predict_graph, init_graph=init_graph)
        onnx_output = c2_ir.run(inputs)
        for blob_name in c2_ref.keys():
            np.testing.assert_almost_equal(onnx_output[blob_name], c2_ref[blob_name], decimal=decimal)

    def _download(self, model):
        model_dir = self.model_dir(model)

        if os.path.exists(model_dir):
            print('Folder {} already exists. Skip download.'.format(model))
            return
        os.makedirs(model_dir)
        for f in ['predict_net.pb', 'predict_net.pbtxt', 'init_net.pb']:
            try:
                downloadFromURLToFile(getURLFromName(model, f),
                                      '{folder}/{f}'.format(folder=model_dir,
                                                            f=f))
            except Exception as e:
                print("Abort: {reason}".format(reason=str(e)))
                print("Cleaning up...")
                deleteDirectory(model_dir)
                exit(0)

    def test_alexnet(self):
        model = 'bvlc_alexnet'
        self._download(model)
        self._test_net(model, decimal=4)

    def test_resnet50(self):
        model = 'resnet50'
        self._download(model)
        self._test_net(model)

    def test_vgg16(self):
        model = 'vgg16'
        self._download(model)
        self._test_net(model)

    def test_vgg19(self):
        model = 'vgg19'
        self._download(model)
        self._test_net(model)

    def test_inception_v1(self):
        model = 'inception_v1'
        self._download(model)
        self._test_net(model, decimal=2)

    def test_inception_v2(self):
        model = 'inception_v2'
        self._download(model)
        self._test_net(model)

    def test_squeezenet(self):
        model = 'squeezenet'
        self._download(model)
        self._test_net(model)

    def test_shufflenet(self):
        model = 'shufflenet'
        self._download(model)
        self._test_net(model)

    def test_densenet121(self):
        model = 'densenet121'
        self._download(model)
        self._test_net(model)


class TestCaffe2ReferenceNode(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=0)


for test in node_tests:
    assert not hasattr(TestCaffe2Reference, test.name), 'Two tests have the same name: ' + test.name
    # Due to Python method binding shenanigans, eta expansion does
    # not hold: (lambda x: m.f(x)) != m.f
    setattr(TestCaffe2ReferenceNode, test.name, lambda self: test.run(self))


if __name__ == '__main__':
    unittest.main()
