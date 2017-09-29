"""Backend for running ONNX on Caffe2

To run this, you will need to have Caffe2 installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib
import uuid
from future.utils import bytes_to_native_str

import caffe2
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import caffe2.python.utils
from onnx import onnx_pb2, checker
from onnx.onnx_pb2 import TensorProto, AttributeProto
import onnx.numpy_helper
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
    def __init__(self, predict_net, device, workspace, uninitialized):
        super(Caffe2Rep, self).__init__()
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
                    assert len(self.uninitialized) == len(inputs), \
                           'Caffe2Rep.Run: length of input must equal to the length of Uninitialized list: {}, \
                            did you initialize the input of the graph in init_graph/initializer?'.format(self.uninitialized)
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


class OnnxAttributes(dict):
    """
    This is a more convenient way to work with ONNX/Caffe2 attributes
    that is not the protobuf representation.
    """
    @staticmethod
    def from_onnx(args):
        d = OnnxAttributes()
        for arg in args:
            d[arg.name] = convertAttributeProto(arg)
        return d
    def caffe2(self, kmap=lambda k: k):
        for k, v in self.items():
            yield caffe2.python.utils.MakeArgument(kmap(k), v)


# TODO: Move this into ONNX main library
def convertAttributeProto(onnx_arg):
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.

    NB: Tensor attribute gets returned as the straight proto.
    """
    if onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        return onnx_arg.t  # this is a proto!
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


# TODO: Move this into ONNX main library
class OnnxNode(object):
    """
    Reimplementation of NodeProto from ONNX, but in a form
    more convenient to work with from Python.

    We may temporarily edit these nodes to get them into Caffe2 form,
    before actually translating into the Caffe2 protobuf, since this
    is easier than decomposing everything, and putting it back together
    when we're ready.
    """
    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.attrs = OnnxAttributes.from_onnx(node.attribute)
        self.consumed_inputs = self.attrs.pop("consumed_inputs", None)
        self.inputs = list(node.input)
        self.outputs = list(node.output)


def translateTensorProto(onnx_tensor, c2_out):
    """
    Given an Onnx TensorProto, translate it into a Caffe2 operator
    which produces the given tensor constant at Caffe2 name c2_out.
    """

    c2_op = caffe2_pb2.OperatorDef()

    c2_values = c2_op.arg.add()
    c2_values.name = "values"

    def tensor2list(onnx_tensor):
        # Use the onnx.helper because the data may be raw
        return onnx.numpy_helper.to_array(onnx_tensor).flatten().tolist()

    if onnx_tensor.data_type == TensorProto.FLOAT:
        c2_op.type = 'GivenTensorFill'
        c2_values.floats.extend(tensor2list(onnx_tensor))
    elif onnx_tensor.data_type == TensorProto.INT64:
        c2_op.type = 'GivenTensorInt64Fill'
        c2_values.ints.extend(tensor2list(onnx_tensor))
    elif onnx_tensor.data_type in [TensorProto.UINT8,
                                   TensorProto.INT8,
                                   TensorProto.UINT16,
                                   TensorProto.INT16,
                                   TensorProto.INT32,
                                   TensorProto.FLOAT16]:
        c2_op.type = 'GivenTensorIntFill'
        c2_values.ints.extend(tensor2list(onnx_tensor))
    elif onnx_tensor.data_type == TensorProto.BOOL:
        c2_op.type = 'GivenTensorBoolFill'
        c2_values.ints.extend(tensor2list(onnx_tensor))
    elif onnx_tensor.data_type == TensorProto.STRING:
        c2_op.type = 'GivenTensorStringFill'
        c2_values.strings.extend(tensor.string_data)
    else:
        raise RuntimeError(
            "unrecognized tensor type {}".format(onnx_tensor.data_type))

    c2_shape = c2_op.arg.add()
    c2_shape.name = "shape"
    c2_shape.ints.extend(onnx_tensor.dims)

    c2_op.output.extend([c2_out])

    return c2_op


class Caffe2Backend(Backend):

    # Operators that are different between Caffe2 and
    # ONNX but only in their name.
    # In most cases, this should be empty - as the effort of ONNX is
    # to unify the operator definitions.
    _renamed_operators = {
        'Dot':                  'MatMul',
        'Caffe2ConvTranspose':  'ConvTranspose',
        'GlobalMaxPool':        'MaxPool',
        'GlobalAveragePool':    'AveragePool',
    }

    _conv_transpose_unpool_renamed_attrs  = {'kernel_shape': 'kernels'}
    _conv_pool_op_renamed_attrs           = {'kernel_shape': 'kernels'}

    # NB: domain is ONNX operator names.  This only really makes
    # sense in the context of _renamed_operators as well
    _renamed_attrs = {
        'Squeeze':              {'axes': 'dims'},
        'Transpose':            {'perm': 'axes'},
        'Caffe2ConvTranspose':  _conv_transpose_unpool_renamed_attrs,
        'Conv':                 _conv_pool_op_renamed_attrs,
        'MaxPool':              _conv_pool_op_renamed_attrs,
        'GlobalMaxPool':        _conv_pool_op_renamed_attrs,
        'AveragePool':          _conv_pool_op_renamed_attrs,
        'GlobalAveragePool':    _conv_pool_op_renamed_attrs,
        'ChannelShuffle':       {'kernel_shape': 'kernels'},
    }

    # operators whose behavior is different beyond renaming
    # the value is an attribute of this class that is a
    # function from ToffeIR node_def to caffe2 op_def
    _special_operators = {
        'Constant': '_create_constant',
        'Conv': '_create_conv_pool_op_base',
        'AveragePool': '_create_conv_pool_op_base',
        'GlobalAveragePool': '_create_conv_pool_op_base',
        'GlobalMaxPool': '_create_conv_pool_op_base',
        'MaxPool': '_create_conv_pool_op_base',
        'Reshape': '_create_reshape',
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
            output_values = [workspace.FetchBlob(env[name]) for name in node.output]
            return namedtupledict('Outputs', node.output)(*output_values)

    @classmethod
    def _create_constant(cls, n, env):
        return translateTensorProto(n.attrs["value"], n.outputs[0])

    # Note [Caffe2 ConvPoolOpBase]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # To understand what is going on here, we have to talk a little bit about
    # Caffe2's internals.
    #
    # First, it's important to know that all of Caffe2's pooling and convolution
    # operators inherit from "ConvPoolOpBase", which is an abstract class that
    # defines all of the attributes (kernels, dilations, strides, etc) which one
    # sees on these operators.  Unfortunately, Caffe2's documentation generator
    # doesn't know how to handle cases like this, so for example, if you look at
    # the docs for MaxPool at <https://caffe2.ai/docs/operators-catalogue.html#maxpool>
    # you won't see any of the attributes.  You have to go source diving to
    # find the information; in particular, you want to look at:
    # https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_pool_op_base.h
    # This class handles *global* pooling as well.
    #
    # Second, it's important to know what Caffe2 expects for padding, which can
    # be somewhat difficult to understand from the code because Caffe2 handles
    # both singular/pluralized spellings of padding, and there is also legacy
    # padding business.  The short version of the story is that, for NON-legacy
    # padding (which is what we want to output), padding is expected to be
    # *twice* the size of kernels.  So if you have a 2D convolution, Caffe2
    # will accept two values in 'kernels', but FOUR values in 'pads';
    # furthermore, this is *mandatory.*
    #
    # Finally, ConvPoolOpBase is not the only class of it's kind; there is
    # also ConvTransposeUnpoolBase, which backs ConvTranspose.  So don't
    # be tricked by the fact that Conv and ConvTranspose have similar
    # parameters; they exercise different codepaths and need to be handled
    # differently.

    @classmethod
    def _create_conv_pool_op_base(cls, n, env):
        if n.op_type.startswith('Global'):
            n.attrs['global_pooling'] = 1

        try:
            kernels = n.attrs['kernel_shape']
            pads = n.attrs['pads']
        except KeyError:
            pass
        else:
            if len(kernels) == len(pads):
                # Caffe2 requires pads to be twice the size of kernels.
                n.attrs['pads'] = pads * 2

        return cls._translate_onnx(n, env)

    @classmethod
    def _create_reshape(cls, n, env):
        c2_op = cls._translate_onnx(n, env)
        # Caffe2 has an extra output
        c2_op.output.extend([env.fresh()])
        return c2_op

    @classmethod
    def prepare(cls, predict_model, device='CPU',
                init_model=None, **kwargs):
        '''
        For Onnx Caffe2Backend, we require that init_graph don't initialize the actual input of the predict_graph,

        for example, if "img" is the input blob for the predict_net, we require that in init_graph and in
        initializer of the predict_graph, "img" is not initalized. We don't have a check for this, since
        there is no way we can know which blob is the input of the predict_graph.
        '''
        super(Caffe2Backend, cls).prepare(predict_model, device, **kwargs)
        if init_model:
            checker.check_model(init_model)

        ws = Workspace()
        device = Device(device)

        predict_net, _ = cls._onnx_graph_to_caffe2_net(predict_model.graph)
        predict_net.device_option.CopyFrom(get_device_option(device))

        with ws, core.DeviceScope(get_device_option(device)):
            for init_tensor in predict_model.graph.initializer:
                workspace.FeedBlob(init_tensor.name, onnx.numpy_helper.to_array(init_tensor))
            if init_model:
                init_net, _ = cls._onnx_graph_to_caffe2_net(init_model.graph)
                workspaces.RunNetOnce(init_net)
            uninitialized = [x
                             for x in predict_net.external_input
                             if not workspace.HasBlob(x)]

        return Caffe2Rep(predict_net, device, ws, uninitialized)

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
            translator = cls._translate_onnx
        return translator(OnnxNode(node_def), env)

    @classmethod
    def _translate_onnx(cls, onnx_node, env):
        """
        This translator performs the basic translation of ONNX nodes into
        Caffe2 operators.  Besides doing a straightforward marshalling from
        one format to another, it also does these extra things:

          - Renames operators based on '_renamed_operators'
          - Renames attributes based on '_renamed_attrs'
          - Handles "consumed_inputs" attribute so that inplace operations are
            encoded correctly in Caffe2.

        If you're writing a custom translator, consider calling this first,
        and then fixing things up further.
        """
        c2_op = caffe2_pb2.OperatorDef()

        c2_op.input.extend([env[i] for i in onnx_node.inputs])

        for output in onnx_node.outputs:
            env[output] = output
        # when consumed_inputs exist, we need to
        # rewrite the outputs to re-use these inputs to
        # support Caffe2-style in-place operators.
        if onnx_node.consumed_inputs:
            schema = onnx.defs.get_schema(onnx_node.op_type)
            for i, input in enumerate(onnx_node.inputs):
                if onnx_node.consumed_inputs[i] != 0:
                    # for each consumed input, the schema for the op
                    # tells us which output (output_idx) that
                    # this consumed input becomes
                    _, output_idx = schema.consumed(i)
                    # consumed outputs are not always present
                    # for instance batch norm in test mode
                    # does not return the consumed inputs
                    if output_idx < len(onnx_node.outputs):
                        # rather than use its ONNX name
                        # use the original input name for the blob
                        # that will be consumed
                        env[onnx_node.outputs[output_idx]] = env[input]

        c2_op.output.extend([env[i] for i in onnx_node.outputs])
        c2_op.name = onnx_node.name
        onnx_op = onnx_node.op_type
        c2_op.type = cls._renamed_operators.get(onnx_op, onnx_op)

        def kmap(k):
            return cls._renamed_attrs.get(onnx_op, {}).get(k, k)
        c2_op.arg.extend(onnx_node.attrs.caffe2(kmap=kmap))
        return c2_op


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


prepare = Caffe2Backend.prepare

run_node = Caffe2Backend.run_node

run_model = Caffe2Backend.run_model
