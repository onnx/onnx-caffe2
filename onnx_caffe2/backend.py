"""Backend for running ONNX on Caffe2

To run this, you will need to have Caffe2 installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib
from future.utils import bytes_to_native_str

import caffe2
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import caffe2.python.utils
from onnx import onnx_pb2, checker
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
import onnx.numpy_helper
import onnx.defs
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict
from onnx_caffe2.workspace import Workspace
from onnx_caffe2.backend_rep import Caffe2Rep
from onnx_caffe2.helper import dummy_name


def get_device_option(device):
    m = {DeviceType.CPU: caffe2_pb2.CPU,
         DeviceType.CUDA: caffe2_pb2.CUDA}
    return core.DeviceOption(m[device.type], device.device_id)


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

    _global_renamed_attrs = {'kernel_shape': 'kernels'}
    _per_op_renamed_attrs = {
        'Squeeze':              {'axes': 'dims'},
        'Transpose':            {'perm': 'axes'},
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

            cls._inplace_rewrite([node])
            workspace.RunOperatorOnce(
                cls._onnx_node_to_caffe2_op(node))
            output_values = [workspace.FetchBlob(name) for name in node.output]
            return namedtupledict('Outputs', node.output)(*output_values)

    @classmethod
    def _create_tensor_filling_op(cls, onnx_tensor, name=None):
        """
        Given an Onnx TensorProto, translate it into a Caffe2 operator
        which produces the given tensor filling op.
        """
        assert name or onnx_tensor.name
        name = name or onnx_tensor.name

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

        c2_op.output.append(name)

        return c2_op

    @classmethod
    def _create_constant(cls, n):
        assert len(n.outputs) == 1
        return cls._create_tensor_filling_op(n.attrs["value"], n.outputs[0])

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
    def _create_conv_pool_op_base(cls, n):
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

        return cls._common_onnx_node_to_caffe2_op(n)

    @classmethod
    def _create_reshape(cls, n):
        c2_op = cls._common_onnx_node_to_caffe2_op(n)
        # Caffe2 has an extra output
        c2_op.output.append(dummy_name())
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

        predict_net = cls.onnx_graph_to_caffe2_net(predict_model.graph)
        predict_net.device_option.CopyFrom(get_device_option(Device(device)))

        ws = Workspace()
        with ws, core.DeviceScope(predict_net.device_option):
            for init_tensor in predict_model.graph.initializer:
                workspace.FeedBlob(init_tensor.name,
                                   onnx.numpy_helper.to_array(init_tensor))
            if init_model:
                init_net = cls.onnx_graph_to_caffe2_net(init_model.graph)
                workspace.RunNetOnce(init_net)
            uninitialized = [x
                             for x in predict_net.external_input
                             if not workspace.HasBlob(x)]

        return Caffe2Rep(predict_net, ws, uninitialized)

    @classmethod
    # TODO: This method needs a refactor for clarity
    def _onnx_node_to_caffe2_op(cls, node_def):
        if node_def.op_type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[node_def.op_type])
        else:
            translator = cls._common_onnx_node_to_caffe2_op
        return translator(OnnxNode(node_def))

    @classmethod
    def _common_onnx_node_to_caffe2_op(cls, onnx_node):
        """
        This translator performs the basic translation of ONNX nodes into
        Caffe2 operators.  Besides doing a straightforward marshalling from
        one format to another, it also does these extra things:

          - Renames operators based on '_renamed_operators'
          - Renames attributes based on '_global_renamed_attrs' and
            '_per_op_renamed_attrs'

        If you're writing a custom translator, consider calling this first,
        and then fixing things up further.
        """
        c2_op = caffe2_pb2.OperatorDef()

        c2_op.input.extend(onnx_node.inputs)
        c2_op.output.extend(onnx_node.outputs)
        c2_op.name = onnx_node.name

        onnx_op_type = onnx_node.op_type
        c2_op.type = cls._renamed_operators.get(onnx_op_type, onnx_op_type)

        def kmap(k):
            if (onnx_op_type in cls._per_op_renamed_attrs and
                k in cls._per_op_renamed_attrs[onnx_op_type]):
                return cls._per_op_renamed_attrs[onnx_op_type][k]
            if k in cls._global_renamed_attrs:
                return cls._global_renamed_attrs[k]
            return k
        c2_op.arg.extend(onnx_node.attrs.caffe2(kmap=kmap))

        return c2_op


    @classmethod
    def _inplace_rewrite(cls, graph_or_nodes):
        '''
        currently we use this to translate ONNX-style
        consumed_input annotations to Caffe2-style in place
        updates (use same input and output names).
        '''
        is_graph = isinstance(graph_or_nodes, GraphProto)
        if is_graph:
            nodes = graph_or_nodes.node
        else:
            nodes = graph_or_nodes

        renamed = {}

        for node in nodes:
            node.input[:] = [renamed.get(input_name, input_name)
                             for input_name in node.input]
            consumed_inputs = OnnxNode(node).consumed_inputs or []
            output_idxes = set(range(len(node.output)))
            schema = onnx.defs.get_schema(node.op_type)
            for i, consumed in enumerate(consumed_inputs):
                if not consumed:
                    continue
                _, output_idx = schema.consumed(i)
                # consumed outputs are not always present
                # for instance batch norm in test mode
                # does not return the consumed inputs
                if output_idx < len(node.output):
                    output_idxes.remove(output_idx)
                    old_val = node.output[output_idx]
                    new_val = node.input[i]
                    node.output[output_idx] = new_val
                    renamed[old_val] = new_val
            for idx in output_idxes:
                name = node.output[idx]
                node.output[idx] = renamed.get(name, name)
        if is_graph:
            for output in graph_or_nodes.output:
                output.name = renamed.get(output.name, output.name)

    @classmethod
    def onnx_graph_to_caffe2_net(cls, graph_def):
        cls._inplace_rewrite(graph_def)

        net_def = caffe2_pb2.NetDef()
        net_def.name = graph_def.name

        net_def.op.extend([cls._onnx_node_to_caffe2_op(node)
                           for node in graph_def.node])
        net_def.external_input.extend(
            value_info.name for value_info in graph_def.input)
        net_def.external_output.extend(
            value_info.name for value_info in graph_def.output)

        return net_def

    @classmethod
    def onnx_initializer_to_caffe2_init_net(cls, initializer, init_net_name='init'):
        net_def = caffe2_pb2.NetDef()
        net_def.name = init_net_name
        net_def.op.extend(cls._create_tensor_filling_op(tp) for tp in initializer)
        return net_def


prepare = Caffe2Backend.prepare

run_node = Caffe2Backend.run_node

run_model = Caffe2Backend.run_model
