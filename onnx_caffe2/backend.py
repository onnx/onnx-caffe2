"""Backend for running ONNX on Caffe2

To run this, you will need to have Caffe2 installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib
import uuid

import caffe2
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from onnx import onnx_pb2, checker
from onnx.onnx_pb2 import TensorProto, AttributeProto
from onnx.numpy_helper import to_array
import onnx.defs

from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict


def get_device_option(device):
    m = {DeviceType.CPU: caffe2_pb2.CPU,
         DeviceType.CUDA: caffe2_pb2.CUDA}
    return core.DeviceOption(m[device.type], device.device_id)

def get_name_scope(device):
    if device.type == DeviceType.CUDA:
        return 'gpu_{}'.format(device.device_id)
    return ''


class Workspace(object):
    """
    An object representing a Caffe2 workspace.  It is a context manager,
    so you can say 'with workspace:' to use the represented workspace
    as your global workspace.  It also supports every method supported
    by caffe2.python.workspace, but instead of running these operations
    in the global workspace, it runs them in the workspace represented
    by this object.  When this object goes dead, the workspace (and all
    nets and blobs within it) are freed.

    Why do we need this class?  Caffe2's workspace model is very "global state"
    oriented, in that there is always some ambient global workspace you are
    working in which holds on to all of your networks and blobs.  This class
    makes it possible to work with workspaces more locally, and without
    forgetting to deallocate everything in the end.
    """
    def __init__(self):
        # Caffe2 (apparently) doesn't provide any native method of generating
        # a fresh, unused workspace, so we have to fake it by generating
        # a unique ID and hoping it's not used already / will not be used
        # directly in the future.
        self.workspace_id = str(uuid.uuid4())
        # A stack, so that the context manager is reentrant.
        self.workspace_stack = []
    def __getattr__(self, attr):
        def f(*args, **kwargs):
            with self:
                return getattr(workspace, attr)(*args, **kwargs)
        return f
    def __enter__(self):
        self.workspace_stack.append(workspace.CurrentWorkspace())
        workspace.SwitchWorkspace(self.workspace_id, create_if_missing=True)
    def __exit__(self, exc_type, exc_value, traceback):
        w = self.workspace_stack.pop()
        # Strictly speaking, create_if_missing here is unnecessary, since a user
        # is not supposed to be allowed to destruct a workspace while we're in
        # it.  However, empirically, it has been observed that during abnormal
        # shutdown, Caffe2 deletes its default workspace fairly early in the
        # final calls to destructors.  In this case, we may attempt to exit
        # to a default workspace which no longer exists.  create_if_missing=True
        # will (harmlessly) recreate the workspace before we finally quit.)
        workspace.SwitchWorkspace(w, create_if_missing=True)
    def __del__(self):
        # NB: This is a 'self' call because we need to switch into the workspace
        # we want to reset before we actually reset it.  A direct call to
        # workspace.ResetWorkspace() will reset the ambient workspace, which
        # is not want we want.
        self.ResetWorkspace()

class Caffe2Rep(BackendRep):
    def __init__(self, init_net, predict_net, device, workspace, uninitialized):
        super(Caffe2Rep, self).__init__()
        self.init_net = init_net
        self.predict_net = predict_net
        self.device = device
        self.workspace = workspace
        # The list of uninitialized external_inputs in workspace, we need this to
        # pair the name with given sequence inputs.
        self.uninitialized = uninitialized
        self.net_created = False

    def run(self, inputs, **kwargs):
        super(Caffe2Rep, self).run(inputs, **kwargs)
        with self.workspace:
            predict_net, device = self.predict_net, self.device
            with core.DeviceScope(get_device_option(device)):
                if isinstance(inputs, dict):
                    with core.NameScope(get_name_scope(device)):
                        for key, value in inputs.items():
                            workspace.FeedBlob(key, value)
                elif isinstance(inputs, list) or isinstance(inputs, tuple):
                    assert len(self.uninitialized) == len(inputs)
                    for i, value in enumerate(inputs):
                        # namescope already baked into protobuf
                        workspace.FeedBlob(self.uninitialized[i], value)
                else:
                    # single input
                    workspace.FeedBlob(self.uninitialized[0], inputs)
                if not self.net_created:
                    workspace.CreateNet(predict_net)
                    self.net_created = True
                workspace.RunNet(predict_net.name)
            output_values = [workspace.FetchBlob(name) for name in predict_net.external_output]
            return namedtupledict('Outputs', predict_net.external_output)(*output_values)


class Caffe2Backend(Backend):

    # Operators that are different between Caffe2 and
    # ONNX but only in their name.
    # In most cases, this should be empty - as the effort of ONNX is
    # to unify the operator definitions.
    _renamed_operators = {
        'Conv1D': 'Conv',
        'Conv2D': 'Conv',
        'Conv3D': 'Conv',
        'MaxPool2D': 'MaxPool',
        'AveragePool2D': 'AveragePool',
    }

    # NB: domain is RENAMED operator names, not the originals
    _renamed_attrs = {
        'Squeeze': {'axes': 'dims'},
        'Transpose': {'perm': 'axes'},
        'Conv': {'kernel_shape': 'kernels'},
        'ConvTranspose': {'kernel_shape': 'kernels'},
        'MaxPool': {'kernel_shape': 'kernels'},
        'AveragePool': {'kernel_shape': 'kernels'},
        'ChannelShuffle': {'kernel_shape': 'kernels'},
    }

    # operators whose behavior is different beyond renaming
    # the value is an attribute of this class that is a
    # function from ToffeIR node_def to caffe2 op_def
    _special_operators = {
        'Constant': '_create_constant',
        'Caffe2ConvTranspose': '_create_transpose',
        'Reshape': '_create_reshape',
        'GlobalAveragePool': '_create_global_pool_op',
        'GlobalMaxPool': '_create_global_pool_op',
    }
    @classmethod
    def run_node(cls, node, inputs):
        super(Caffe2Backend, cls).run_node(node, inputs)

        with Workspace(): # temporary!
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    workspace.FeedBlob(key, value)
            else:
                assert(len(node.input) == len(inputs))
                for key, value in zip(node.input, inputs):
                    workspace.FeedBlob(key, value)
            env = {}
            for input in node.input:
                env[input] = input
            workspace.RunOperatorOnce(
                cls._onnx_node_to_caffe2_op(node, env))
            return dict(
                (name, workspace.FetchBlob(env[name]))
                for name in node.output)

    @classmethod
    def fill_values(cls, arg, tensor):
        data_type = tensor.data_type
        if data_type == TensorProto.STRING:
            arg.strings.extend(tensor.string_data)
            return

        nptensor_data = to_array(tensor).flatten().tolist()
        if data_type == TensorProto.FLOAT:
            arg.floats.extend(nptensor_data)
        elif data_type in [TensorProto.UINT8,
                           TensorProto.INT8,
                           TensorProto.UINT16,
                           TensorProto.INT16,
                           TensorProto.INT32,
                           TensorProto.FLOAT16,
                           TensorProto.BOOL,
                           TensorProto.INT64]:
            # TODO: Using this for FLOAT16 seems questionable
            arg.ints.extend(nptensor_data)

    @classmethod
    def _get_attribute_by_name(cls, node_def, name):
        for attr in node_def.attribute:
            if attr.name == name:
                return attr
        raise IndexError('onnx node has no attribute ' + name)

    @classmethod
    def _create_constant(cls, node_def, env):
        op_def = caffe2_pb2.OperatorDef()
        op_def.output.extend([env[o] for o in node_def.output])

        init_tensor = cls._get_attribute_by_name(node_def, 'value').t
        if init_tensor.data_type == TensorProto.FLOAT:
            op_def.type = 'GivenTensorFill'
        elif init_tensor.data_type == TensorProto.INT64:
            op_def.type = 'GivenTensorInt64Fill'
        elif init_tensor.data_type == TensorProto.INT32:
            op_def.type = 'GivenTensorIntFill'
        elif init_tensor.data_type == TensorProto.BOOL:
            op_def.type = 'GivenTensorBoolFill'
        elif init_tensor.data_type == TensorProto.STRING:
            op_def.type = 'GivenTensorStringFill'
        else:
            raise RuntimeError("unrecognized tensor constant type {}".format(init_tensor.data_type))

        values = op_def.arg.add()
        values.name = "values"

        cls.fill_values(values, init_tensor)
        shape = op_def.arg.add()
        shape.name = "shape"
        shape.ints.extend(init_tensor.dims)

        return op_def

    @classmethod
    def _create_transpose(cls, node_def, env):
        op_def = caffe2_pb2.OperatorDef()
        op_def.output.extend([env[o] for o in node_def.output])
        op_def.input.extend([env[i] for i in node_def.input])
        op_def.type = 'ConvTranspose'
        op_def.name = node_def.name

        def can_be_singular(values):
            if len(values) == 0:
                return False
            return all(values[0] == v for v in values)

        depluralizer = { 'kernel_shape': 'kernel', 'strides': 'stride', 'pads': 'pad' }
        def map_attr(attr):
            if attr.name in depluralizer:
                # TODO: replace this with a version test
                if not can_be_singular(attr.ints):
                    raise "Caffe2 doesn't support plural kernel_shape/strides/pads prior to 6cb4d1ecb0dfb553f797f6a8a61dd6966909cb0b; if you know your Caffe2 is recent enough, comment out this test"
                # In fact, this code is MANDATORY, because prior to
                # https://github.com/caffe2/caffe2/commit/6cb4d1ecb0dfb553f797f6a8a61dd6966909cb0b
                # the pluralized versions were not supported.
                # You'll get an error like
                # "[enforce fail at conv_transpose_unpool_op_base.h:54] kernel_h_ > 0"
                # if your Caffe2 is too old and you actually use the plural
                # version
                singular_attr = AttributeProto()
                singular_attr.name = depluralizer[attr.name]
                singular_attr.i = attr.ints[0]
                return cls._onnx_arg_to_caffe2_arg(op_def.type, singular_attr)
            else:
                return cls._onnx_arg_to_caffe2_arg(op_def.type, attr)

        op_def.arg.extend([map_attr(attr) for attr in node_def.attribute])
        return op_def

    @classmethod
    def _create_reshape(cls, node_def, env):
        op_def = caffe2_pb2.OperatorDef()
        # Caffe2 has an extra output
        op_def.output.extend([env[o] for o in node_def.output] + [env.fresh()])
        op_def.input.extend([env[i] for i in node_def.input])
        op_def.type = 'Reshape'
        op_def.name = node_def.name
        op_def.arg.extend(map(lambda x: cls._onnx_arg_to_caffe2_arg(op_def.type, x), node_def.attribute))
        return op_def

    @classmethod
    def _create_global_pool_op(cls, node_def, env):
        op_def = cls._common_op_tranlator(node_def, env)

        assert node_def.op_type.startswith('Global')
        op_def.type = node_def.op_type.split('Global')[-1]

        global_pooling_attr = op_def.arg.add()
        global_pooling_attr.name = 'global_pooling'
        global_pooling_attr.i = 1

        return op_def

    @classmethod
    def prepare(cls, predict_graph, device='CPU',
                init_graph=None, **kwargs):
        super(Caffe2Backend, cls).prepare(predict_graph, device, **kwargs)
        if init_graph:
            checker.check_graph(init_graph)

        tmp_ws = Workspace()

        with tmp_ws:
            device = Device(device)

            if init_graph:
                init_net, _ = cls._onnx_graph_to_caffe2_net(init_graph)
            else:
                init_net = caffe2_pb2.NetDef()

            predict_net, _ = cls._onnx_graph_to_caffe2_net(predict_graph)
            predict_net.device_option.CopyFrom(get_device_option(device))

            with core.DeviceScope(get_device_option(device)):
                for init_tensor in predict_graph.initializer:
                    workspace.FeedBlob(init_tensor.name, to_array(init_tensor))
                workspace.RunNetOnce(init_net)
            uninitialized = filter(lambda x:not workspace.HasBlob(x), predict_net.external_input)
            return Caffe2Rep(init_net, predict_net, device, tmp_ws, uninitialized)

    @classmethod
    def run_graph(cls, predict_graph, inputs, device='CPU',
                  init_graph=None,
                  **kwargs):
        super(Caffe2Backend, cls).run_graph(predict_graph, inputs, device, **kwargs)
        c2_rep = cls.prepare(predict_graph, device, init_graph)
        return c2_rep.run(inputs)

    @classmethod
    def _onnx_arg_to_caffe2_arg(cls, op_name, onnx_arg):
        c2_arg = caffe2_pb2.Argument()
        if op_name in cls._renamed_attrs and onnx_arg.name in cls._renamed_attrs[op_name]:
            # Handle renamed attributes
            c2_arg.name = cls._renamed_attrs[op_name][onnx_arg.name]
        else:
            c2_arg.name = onnx_arg.name
        if onnx_arg.HasField('f'):
            c2_arg.f = onnx_arg.f
        elif onnx_arg.HasField('i'):
            c2_arg.i = onnx_arg.i
        elif onnx_arg.HasField('s'):
            c2_arg.s = onnx_arg.s
        elif len(onnx_arg.floats):
            c2_arg.floats.extend(onnx_arg.floats)
        elif len(onnx_arg.ints):
            c2_arg.ints.extend(onnx_arg.ints)
        elif len(onnx_arg.strings):
            c2_arg.strings.extend(onnx_arg.strings)
        elif onnx_arg.HasField('graphs'):
            raise NotImplementedError(
                "Caffe2 backend does not support sub graph yet.")
        return c2_arg

    @classmethod
    # TODO: This method needs a refactor for clarity
    def _onnx_node_to_caffe2_op(cls, node_def, env):
        # This needs to be done for special operators and regular ones
        for output in node_def.output:
            if output not in env:
                env[output] = output

        if node_def.op_type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[node_def.op_type])
        else:
            translator = cls._common_op_tranlator
        return translator(node_def, env)

    @classmethod
    def _common_op_tranlator(cls, node_def, env):
        op_def = caffe2_pb2.OperatorDef()
        op_def.input.extend([env[i] for i in node_def.input])

        for output in node_def.output:
            env[output] = output

        # when consumed_inputs exist, we need to
        # rewrite the outputs to re-use these inputs to
        # support Caffe2-style in-place operators.
        for attr in node_def.attribute:
            if attr.name == "consumed_inputs":
                schema = onnx.defs.get_schema(node_def.op_type)
                for i,input in enumerate(node_def.input):
                    if attr.ints[i] != 0:
                        # for each consumed input, the schema for the op
                        # tells us which output (output_idx) that
                        # this consumed input becomes
                        _, output_idx = schema.consumed(i)
                        # consumed outputs are not always present
                        # for instance batch norm in test mode
                        # does not return the consumed inputs
                        if output_idx < len(node_def.output):
                            # rather than use its ONNX name
                            # use the original input name for the blob
                            # that will be consumed
                            env[node_def.output[output_idx]] = env[input]

        op_def.output.extend([env[i] for i in node_def.output])
        op_def.name = node_def.name
        op_def.type = cls._renamed_operators.get(node_def.op_type, node_def.op_type)
        op_def.arg.extend(
            cls._onnx_arg_to_caffe2_arg(op_def.type, a) for a in node_def.attribute
            if a.name != "consumed_inputs")
        return op_def


    @classmethod
    def _onnx_graph_to_caffe2_net(cls, graph_def):

        # This is a hotfix to handle caffe2 frontend which sometimes
        # creates ONNX graphs with edges which are not declared anywhere.
        # See resnet50 for an example.
        class RenameEnv(dict):
            def __init__(self):
                self.unique = 0
            def __missing__(self, key):
                return key
            def fresh(self):
                self.unique += 1
                # TODO: Make this robust against an adversarial model namer
                return "_onnx_dummy" + str(self.unique)

        net_def = caffe2_pb2.NetDef()
        net_def.name = graph_def.name

        # environment from ONNX name to Caffe2 name
        # currently we use this to translate ONNX-style
        # consumed_input annotations to Caffe2-style in place
        # updates with repeated input/output names
        env = RenameEnv()
        for input in graph_def.input:
            env[input] = input
        net_def.op.extend([cls._onnx_node_to_caffe2_op(node, env)
                           for node in graph_def.node])
        net_def.external_input.extend(graph_def.input)
        net_def.external_output.extend([env[o] for o in graph_def.output])
        return net_def, env


run_node = Caffe2Backend.run_node

prepare = Caffe2Backend.prepare

run_graph = Caffe2Backend.run_graph
