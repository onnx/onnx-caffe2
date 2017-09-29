from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from caffe2.proto import caffe2_pb2
from caffe2.python import brew
from caffe2.python.model_helper import ModelHelper
from click.testing import CliRunner
import numpy as np
from onnx import onnx_pb2, helper
from onnx_caffe2.helper import make_model

from onnx_caffe2.bin.conversion import caffe2_to_onnx, onnx_to_caffe2
from test_utils import TestCase


class TestConversion(TestCase):
    def _run_command(self, cmd, *args, **kwargs):
        runner = CliRunner()
        result = runner.invoke(cmd, *args, **kwargs)
        self.assertEqual(result.exit_code, 0, result.output)
        assert not result.exception
        return result

    def test_caffe2_to_onnx(self):
        caffe2_net = tempfile.NamedTemporaryFile()
        caffe2_init_net = tempfile.NamedTemporaryFile()
        output = tempfile.NamedTemporaryFile()

        model = ModelHelper(name='caffe2-to-onnx-test')
        brew.relu(model, ["X"], "Y")
        caffe2_net.write(model.net.Proto().SerializeToString())
        caffe2_net.flush()

        init_model = ModelHelper(name='caffe2-to-onnx-init-test')
        init_model.net.GivenTensorFill([], 'X', shape=[2, 2],
                                       values=np.zeros((2, 2)).flatten().astype(float))
        caffe2_init_net.write(init_model.net.Proto().SerializeToString())
        caffe2_init_net.flush()

        result = self._run_command(
            caffe2_to_onnx, [
                '--caffe2-net', caffe2_net.name,
                '--caffe2-init-net', caffe2_init_net.name,
                '--output', output.name,
            ])

        onnx_model = onnx_pb2.ModelProto()
        onnx_model.ParseFromString(output.read())
        self.assertEqual(len(onnx_model.graph.node), 1)
        self.assertEqual(onnx_model.graph.node[0].op_type, 'Relu')
        self.assertEqual(len(onnx_model.graph.initializer), 1)
        self.assertEqual(onnx_model.graph.initializer[0].name, 'X')

    def test_onnx_to_caffe2(self):
        onnx_model = tempfile.NamedTemporaryFile()
        output = tempfile.NamedTemporaryFile()
        init_net_output = tempfile.NamedTemporaryFile()

        node_def = helper.make_node(
            "Relu", ["X"], ["Y"])
        graph_def = helper.make_graph(
            [node_def],
            "test",
            ["X"],
            ["Y"],
            initializer=[helper.make_tensor("X",
                                            onnx_pb2.TensorProto.FLOAT,
                                            [2, 2],
                                            np.zeros((2, 2)).flatten().astype(float))])
        model_def = make_model(graph_def, producer_name='onnx-to-caffe2-test')
        onnx_model.write(model_def.SerializeToString())
        onnx_model.flush()

        result = self._run_command(
            onnx_to_caffe2, [
                '--onnx-model', onnx_model.name,
                '--output', output.name,
                '--init-net-output', init_net_output.name,
            ])

        caffe2_net = caffe2_pb2.NetDef()
        caffe2_net.ParseFromString(output.read())
        self.assertEqual(len(caffe2_net.op), 1)
        self.assertEqual(caffe2_net.op[0].type, 'Relu')

        caffe2_init_net = caffe2_pb2.NetDef()
        caffe2_init_net.ParseFromString(init_net_output.read())
        self.assertEqual(len(caffe2_init_net.op), 1)
