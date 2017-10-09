from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import unittest

from caffe2.proto import caffe2_pb2

import onnx
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info
from onnx_caffe2.helper import make_model, c2_native_run_net

from onnx import onnx_pb2
import onnx_caffe2.frontend as c2_onnx
import onnx_caffe2.backend as c2

import numpy as np
import google.protobuf.text_format
from caffe2.python.models.download import downloadFromURLToFile, getURLFromName, deleteDirectory

from onnx_caffe2.helper import make_model, dummy_name
from test_utils import TestCase


class TestCaffe2Basic(TestCase):
    def test_dummy_name(self):
        n1 = dummy_name()
        n2 = dummy_name()
        assert n1 != n2, "Got same names in different calls: {}".format(n1)

    def test_relu_node_inplace(self):
        X = np.random.randn(3, 2).astype(np.float32)
        Y_ref = np.clip(X, 0, np.inf)

        node_def = make_node(
            "Relu", ["X"], ["X"], consumed_inputs=[1])
        output = c2.run_node(
            node_def, {"X": X})
        np.testing.assert_almost_equal(output.X, Y_ref)

        graph_def = make_graph(
            [node_def],
            name="test",
            inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 2])],
            outputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 2])])
        c2_rep = c2.prepare(make_model(graph_def))
        output = c2_rep.run({"X": X})
        np.testing.assert_almost_equal(output.X, Y_ref)

    def test_relu_graph(self):
        X = np.random.randn(3, 2).astype(np.float32)
        Y_ref = np.clip(X, 0, np.inf)

        node_def = make_node(
            "Relu", ["X"], ["Y"])
        output = c2.run_node(
            node_def, {"X": X})
        np.testing.assert_almost_equal(output.Y, Y_ref)

        graph_def = make_graph(
            [node_def],
            name="test",
            inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 2])],
            outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 2])])
        c2_rep = c2.prepare(make_model(graph_def))
        output = c2_rep.run(X)
        np.testing.assert_almost_equal(output.Y, Y_ref)

    def test_initializer(self):
        X = np.array([[1, 2], [3, 4]]).astype(np.float32)
        Y = np.array([[1, 2], [3, 4]]).astype(np.float32)
        weight = np.array([[1, 0], [0, 1]])
        graph_def = make_graph(
            [make_node("Add", ["X", "Y"], ["Z0"]),
             make_node("Cast", ["Z0"], ["Z"], to="float"),
             make_node("Mul", ["Z", "weight"], ["W"]),
             make_node("Tanh", ["W"], ["W"]),
             make_node("Sigmoid", ["W"], ["W"]),
             make_node("Scale", ["W"], ["W"], scale=-1.0)],
            name="test_initializer",
            inputs=[
                make_tensor_value_info("X", onnx.TensorProto.FLOAT, (2, 2)),
                make_tensor_value_info("Y", onnx.TensorProto.FLOAT, (2, 2)),
                make_tensor_value_info("weight", onnx.TensorProto.FLOAT, (2, 2)),
            ],
            outputs=[
                make_tensor_value_info("W", onnx.TensorProto.FLOAT, (2, 2))
            ],
            initializer=[make_tensor("weight",
                                     onnx_pb2.TensorProto.FLOAT,
                                     [2, 2],
                                     weight.flatten().astype(float))]
        )

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        W_ref = -sigmoid(np.tanh((X + Y) * weight))
        c2_rep = c2.prepare(make_model(graph_def))
        output = c2_rep.run({"X": X, "Y": Y})
        np.testing.assert_almost_equal(output["W"], W_ref)

    def test_gemm(self):
        # simple
        A = np.random.randn(3, 2).astype(np.float32)
        B = np.random.randn(2, 4).astype(np.float32)
        C = np.random.randn(3, 4).astype(np.float32)
        node_def = make_node(
            'Gemm',
            ['A', 'B', 'C'],
            ["Y"])
        output = c2.run_node(node_def, [A, B, C])
        np.testing.assert_almost_equal(output["Y"], np.dot(A, B) + C)

        # transA
        A = np.transpose(A)
        node_def = make_node(
            'Gemm',
            ['A', 'B', 'C'],
            ["Y"],
            transA=True)
        output = c2.run_node(node_def, [A, B, C])
        np.testing.assert_almost_equal(
            output["Y"],
            np.dot(np.transpose(A), B) + C)
        # revert A
        A = np.transpose(A)

        # transB
        B = np.transpose(B)
        node_def = make_node(
            'Gemm',
            ['A', 'B', 'C'],
            ["Y"],
            transB=True)
        output = c2.run_node(node_def, [A, B, C])
        np.testing.assert_almost_equal(
            output["Y"],
            np.dot(A, np.transpose(B)) + C)
        # revert A
        B = np.transpose(B)

        # scale
        alpha = np.random.random()
        beta = np.random.random()
        node_def = make_node(
            'Gemm',
            ['A', 'B', 'C'],
            ["Y"],
            alpha=alpha,
            beta=beta)
        output = c2.run_node(node_def, [A, B, C])
        np.testing.assert_almost_equal(
            output["Y"],
            alpha * np.dot(A, B) + beta * C)

        # broadcast
        C = np.random.randn(4).astype(np.float32)
        node_def = make_node(
            'Gemm',
            ['A', 'B', 'C'],
            ["Y"],
            alpha=alpha,
            beta=beta,
            broadcast=1)
        output = c2.run_node(node_def, [A, B, C])
        np.testing.assert_almost_equal(
            output["Y"],
            alpha * np.dot(A, B) + beta * C)


class TestCaffe2End2End(TestCase):
    def _model_dir(self, model):
        caffe2_home = os.path.expanduser(os.getenv('CAFFE2_HOME', '~/.caffe2'))
        models_dir = os.getenv('CAFFE2_MODELS', os.path.join(caffe2_home, 'models'))
        return os.path.join(models_dir, model)

    def _test_net(self,
                  net_name,
                  input_blob_dims=(1, 3, 224, 224),
                  decimal=7):
        np.random.seed(seed=0)
        model_dir = self._model_dir(net_name)
        if not os.path.exists(model_dir):
            self._download(net_name)
        c2_predict_pb = os.path.join(model_dir, 'predict_net.pb')
        c2_predict_net = caffe2_pb2.NetDef()
        with open(c2_predict_pb, 'rb') as f:
            c2_predict_net.ParseFromString(f.read())
        c2_predict_net.name = net_name

        c2_init_pb = os.path.join(model_dir, 'init_net.pb')
        c2_init_net = caffe2_pb2.NetDef()
        with open(c2_init_pb, 'rb') as f:
            c2_init_net.ParseFromString(f.read())
        c2_init_net.name = net_name + '_init'

        n, c, h, w = input_blob_dims
        data = np.random.randn(n, c, h, w).astype(np.float32)
        inputs = [data]
        c2_outputs = c2_native_run_net(c2_init_net, c2_predict_net, inputs)

        for input_name in c2_predict_net.external_input:
            if input_name == 'data' or input_name == 'gpu_0/data':
                break
        else:
            raise RuntimeError(
                'Could not find input name of model {}'.format(net_name))
        predict_model = c2_onnx.caffe2_net_to_onnx_model(
            predict_net=c2_predict_net,
            init_net=c2_init_net,
            value_info={
                input_name: (onnx_pb2.TensorProto.FLOAT,
                             (1, 3, 224, 224))
            })
        c2_ir = c2.prepare(predict_model)
        onnx_outputs = c2_ir.run(inputs)
        self.assertSameOutputs(c2_outputs, onnx_outputs, decimal=decimal)
        self.report_mem_usage(net_name)

    def _download(self, model):
        model_dir = self._model_dir(model)
        assert not os.path.exists(model_dir)
        os.makedirs(model_dir)
        for f in ['predict_net.pb', 'predict_net.pbtxt', 'init_net.pb']:
            url = getURLFromName(model, f)
            dest = os.path.join(model_dir, f)
            try:
                try:
                    downloadFromURLToFile(url, dest,
                                          show_progress=False)
                except TypeError:
                    # show_progress not supported prior to
                    # Caffe2 78c014e752a374d905ecfb465d44fa16e02a28f1
                    # (Sep 17, 2017)
                    downloadFromURLToFile(url, dest)
            except Exception as e:
                print("Abort: {reason}".format(reason=e))
                print("Cleaning up...")
                deleteDirectory(model_dir)
                exit(1)

    def test_alexnet(self):
        self._test_net('bvlc_alexnet', decimal=4)

    def test_resnet50(self):
        self._test_net('resnet50')

    def test_vgg16(self):
        self._test_net('vgg16')

    @unittest.skipIf(
        os.environ.get('TRAVIS'),
        'Running vgg19 on Travis with Python 2 keeps getting OOM!')
    def test_vgg19(self):
        self._test_net('vgg19')

    def test_inception_v1(self):
        self._test_net('inception_v1', decimal=2)

    def test_inception_v2(self):
        self._test_net('inception_v2')

    def test_squeezenet(self):
        self._test_net('squeezenet')

    def test_shufflenet(self):
        self._test_net('shufflenet')

    def test_densenet121(self):
        self._test_net('densenet121')


if __name__ == '__main__':
    unittest.main()
