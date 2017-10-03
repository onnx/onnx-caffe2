from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.proto import caffe2_pb2
import click
import numpy as np
from onnx import checker, onnx_pb2

from onnx_caffe2.backend import Caffe2Backend as c2
import onnx_caffe2.frontend as c2_onnx


@click.command(
    help='convert caffe2 net to onnx model',
    context_settings={
        'help_option_names': ['-h', '--help']
    }
)
@click.argument('caffe2_net', type=click.File('rb'))
@click.option('--caffe2-net-name',
              type=str,
              help="Name of the caffe2 net")
@click.option('--caffe2-init-net',
              type=click.File('rb'),
              help="Path of the caffe2 init net pb file")
@click.option('-o', '--output', required=True,
              type=click.File('wb'),
              help='Output path for the onnx model pb file')
def caffe2_to_onnx(caffe2_net,
                   caffe2_net_name,
                   caffe2_init_net,
                   output):
    c2_net_proto = caffe2_pb2.NetDef()
    c2_net_proto.ParseFromString(caffe2_net.read())
    if not c2_net_proto.name and not caffe2_net_name:
        raise click.BadParameter(
            'The input caffe2 net does not have name, '
            '--caffe2-net-name must be provided')
    c2_net_proto.name = caffe2_net_name or c2_net_proto.name
    onnx_model = c2_onnx.caffe2_net_to_onnx_model(c2_net_proto)

    if caffe2_init_net:
        c2_init_net_proto = caffe2_pb2.NetDef()
        c2_init_net_proto.ParseFromString(caffe2_init_net.read())
        c2_init_net_proto.name = '{}_init'.format(caffe2_net_name)
        initializer = c2_onnx.caffe2_init_net_to_initializer(c2_init_net_proto)
        onnx_model.graph.initializer.extend(initializer)

    output.write(onnx_model.SerializeToString())


@click.command(
    help='convert onnx model to caffe2 net',
    context_settings={
        'help_option_names': ['-h', '--help']
    }
)
@click.argument('onnx_model', type=click.File('rb'))
@click.option('-o', '--output', required=True,
              type=click.File('wb'),
              help='Output path for the caffe2 net file')
@click.option('--init-net-output',
              type=click.File('wb'),
              help='Output path for the caffe2 init net file')
def onnx_to_caffe2(onnx_model, output, init_net_output):
    onnx_model_proto = onnx_pb2.ModelProto()
    onnx_model_proto.ParseFromString(onnx_model.read())
    graph_def = onnx_model_proto.graph

    if graph_def.initializer:
        if not init_net_output:
            raise click.BadParameter(
                'The input onnx model has initializer, '
                '--init-net-output must be provided '
                'for creating a separated caffe2 init net')
        init_net = c2.onnx_initializer_to_caffe2_init_net(
            graph_def.initializer)
        init_net_output.write(init_net.SerializeToString())
        del graph_def.initializer[:]

    caffe2_net = c2.onnx_graph_to_caffe2_net(graph_def)
    output.write(caffe2_net.SerializeToString())
