"""Caffe2 Protobuf to ONNX converter

To run this, you will need to have Caffe2 installed as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import re

import caffe2
from onnx import onnx_pb2, checker, numpy_helper, mapping
import onnx.helper
from onnx_caffe2.helper import make_model, c2_native_run_net
import onnx.defs
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# caffe2 arguments that needs to be removed
_blacklist_caffe2_args = {'order', 'global_pooling',
                          'cudnn_exhaustive_search', 'use_cudnn'}

# expected argument values
_expected_arg_values = {'order': [b'NCHW'], 'global_pooling': [1]}

_renamed_args = {
    'Squeeze': {'dims': 'axes'},
    'Transpose': {'axes': 'perm'},
    'Conv': {'kernels': 'kernel_shape'},
    'ConvTranspose': {'kernels': 'kernel_shape'},
    'MaxPool': {'kernels': 'kernel_shape'},
    'AveragePool': {'kernels': 'kernel_shape'},
    'ChannelShuffle': {'kernels': 'kernel_shape'},
}


class ArgType(Enum):
    NONE = 0
    FLOAT = 1
    INT = 2
    STR = 3
    FLOATS = 4
    INTS = 5
    STRS = 6


def get_caffe2_arg_type_and_val(caffe2_arg):
    if caffe2_arg.HasField('f'):
        return ArgType.FLOAT, caffe2_arg.f
    elif caffe2_arg.HasField('i'):
        return ArgType.INT, caffe2_arg.i
    elif caffe2_arg.HasField('s'):
        return ArgType.STR, caffe2_arg.s
    elif len(caffe2_arg.floats):
        return ArgType.FLOATS, caffe2_arg.floats
    elif len(caffe2_arg.ints):
        return ArgType.INTS, caffe2_arg.ints
    elif len(caffe2_arg.strings):
        return ArgType.STRS, caffe2_arg.strings
    return ArgType.NONE, None


def onnx_attr_assign(onnx_attr, attr_type, attr_val):
    if attr_type == ArgType.FLOAT:
        onnx_attr.f = attr_val
    elif attr_type == ArgType.INT:
        onnx_attr.i = attr_val
    elif attr_type == ArgType.STR:
        onnx_attr.s = attr_val
    elif attr_type == ArgType.FLOATS:
        onnx_attr.floats.extend(attr_val)
    elif attr_type == ArgType.INTS:
        onnx_attr.ints.extend(attr_val)
    elif attr_type == ArgType.STRS:
        onnx_attr.strings.extend(attr_val)


def get_onnx_attrs(op_type, op_def):
    onnx_attrs = []
    args = {a.name: get_caffe2_arg_type_and_val(a) for a in op_def.arg}
    if op_type in ['Conv',
                   'ConvTranspose',
                   'MaxPool', 'GlobalMaxPool',
                   'AveragePool', 'GlobalAveragePool',
                   'ChannelShuffle']:
        def apply_trans(args, k, dim=2):
            onnx_attr = None
            if dim == 2:
                k_h, k_w, ks = k+'_h', k+'_w', k+'s'
            else:
                k_t, k_l, k_b, k_r, ks = k+'_t', k+'_l', k+'_b', k+'_r', k+'s'
            if dim == 2 and k_h in args and k_w in args:
                assert not onnx_attr
                onnx_attr = onnx_pb2.AttributeProto()
                onnx_attr.name = ks
                onnx_attr_assign(onnx_attr, ArgType.INTS, [args[k_h][1], args[k_w][1]])
                del args[k_h]
                del args[k_w]
            if dim == 4 and k_t in args and k_l in args and k_b in args and k_r in args:
                assert not onnx_attr
                onnx_attr = onnx_pb2.AttributeProto()
                onnx_attr.name = ks
                onnx_attr_assign(onnx_attr, ArgType.INTS, [args[k_t][1], args[k_l][1], args[k_b][1], args[k_r][1]])
                del args[k_t]
                del args[k_l]
                del args[k_b]
                del args[k_r]
            if k in args:
                assert not onnx_attr
                onnx_attr = onnx_pb2.AttributeProto()
                onnx_attr.name = ks
                onnx_attr_assign(onnx_attr, ArgType.INTS, [args[k][1]] * dim)
                del args[k]
            if onnx_attr:
                if op_type in ['GlobalMaxPool', 'GlobalAveragePool']:
                    # TODO: check the values are equal to the default values in c2
                    pass
                else:
                    onnx_attrs.append(onnx_attr)

        apply_trans(args, 'kernel')
        apply_trans(args, 'stride')
        apply_trans(args, 'dilation')
        apply_trans(args, 'adj')
        apply_trans(args, 'pad', 4)

    for a in args:
        t, val = args[a]
        if a in _expected_arg_values:
            if val not in _expected_arg_values[a]:
                raise Exception('value {} not in the expected value list({})'
                                'for argument {}'.format(val, _expected_arg_values[a], a))
        if a not in _blacklist_caffe2_args:
            onnx_attr = onnx_pb2.AttributeProto()
            onnx_attr.name = a
            onnx_attr_assign(onnx_attr, t, val)
            onnx_attrs.append(onnx_attr)

    for attr in onnx_attrs:
        if op_type in _renamed_args and attr.name in _renamed_args[op_type]:
            attr.name = _renamed_args[op_type][attr.name]
    return onnx_attrs


def get_node_op_type(op_def):
    op_type = op_def.type

    matched = re.match(r'^(Conv|ConvTranspose)(\dD)?$', op_type)
    if matched:
        op_type = matched.group(1)

    matched = re.match(r'^(MaxPool|AveragePool)(\dD)?$', op_type)
    if matched:
        is_global = False
        for arg in op_def.arg:
            if arg.name == 'global_pooling' and arg.i:
                is_global = True
                break
        op_type = ('Global' if is_global else '') + matched.group(1)

    return op_type


def caffe2_op_to_node_def(op_def, env):
    node_def = onnx_pb2.NodeProto()
    # NB: This must happen BEFORE we start freshening inplace outputs
    node_def.input.extend(map(env.rename, op_def.input))
    node_def.op_type = get_node_op_type(op_def)

    # Determine what was inplace updates
    input_set = set(op_def.input)
    output_set = set(op_def.output)

    schema = onnx.defs.get_schema(node_def.op_type)
    # ints does not support extend()
    consumes = []
    for i, x in enumerate(op_def.input):
        is_consumed, output_idx = schema.consumed(i)
        if is_consumed == onnx.defs.OpSchema.UseType.CONSUME_ENFORCED:
            consumes.append(1)
        elif is_consumed == onnx.defs.OpSchema.UseType.CONSUME_ALLOWED:
            if x in output_set:
                consumes.append(1)
            else:
                consumes.append(0)
        else:
            if x in output_set:
                raise RuntimeError(
                    "schema says consume not allowed, but caffe2 used inplace syntax")
            consumes.append(0)
    if any(consumes):
        consumes_attr = onnx_pb2.AttributeProto()
        consumes_attr.name = "consumed_inputs"
        consumes_attr.ints.extend(consumes)
    else:
        consumes_attr = None

    def fresh_or_rename(out):
        if out in input_set:
            return env.fresh(out)
        else:
            return env.rename(out)

    node_def.output.extend(map(fresh_or_rename, op_def.output))
    # TODO: refactor frontend to allow special handling for individual ops
    if node_def.op_type == 'Concat':
        assert len(node_def.output) == 2
        del node_def.output[1]

    node_def.name = op_def.name
    attrs = get_onnx_attrs(node_def.op_type, op_def)
    if consumes_attr:
        attrs.append(consumes_attr)
    node_def.attribute.extend(attrs)
    checker.check_node(node_def)
    return node_def


class NameMap(object):
    """
    A class for handling renaming of edges from Caffe2 to ONNX.
    As much as possible, this prefers using the original name,
    but we may need to allocate fresh names to disambiguate inplace
    outputs.
    """
    def __init__(self):
        # Invariant: used = rng(env)
        self.env = {}
        self.used = set()
        self.unique = 0
    def _add(self, k, v):
        self.env[k] = v
        self.used.add(v)
        return v
    def rename(self, name):
        if name in self.env:
            return self.env[name]
        elif name in self.used:
            return self.fresh(name)
        else:
            return self._add(name, name)
    def fresh(self, name):
        def mk():
            return name + "_fresh" + str(self.unique)
        while mk() in self.used:
            self.unique += 1
        # allowed to override!
        return self._add(name, mk())


def caffe2_init_net_to_initializer(init_net):
    initializer = []
    for init_op in init_net.op:
        assert not init_op.input
        try:
            data_type, field_name, np_type, raw = {
                'GivenTensorFill': (onnx_pb2.TensorProto.FLOAT, 'floats', np.float32, True),
                'GivenTensorInt64Fill': (onnx_pb2.TensorProto.INT64, 'ints', np.int64, True),
                'GivenTensorIntFill': (onnx_pb2.TensorProto.INT32, 'ints', np.int32, True),
                'GivenTensorBoolFill': (onnx_pb2.TensorProto.BOOL, 'ints', np.int32, True),
                # raw_data can not be used to store strings
                'GivenTensorStringFill': (onnx_pb2.TensorProto.STRING, 'strings', None, False),
            }[init_op.type]
        except KeyError:
            raise RuntimeError(
                "Can not translate init_net with operator '{}' "
                "to initializer".format(init_op.type)
            )
        args = {a.name: a for a in init_op.arg}
        vals = getattr(args['values'], field_name)
        if raw:
            assert np_type
            vals = np.asarray(vals, dtype=np_type).tobytes()
        initializer.append(onnx.helper.make_tensor(
            name=init_op.output[0],
            data_type=data_type,
            dims=args['shape'].ints,
            vals=vals,
            raw=raw,
        ))
    return initializer


def caffe2_net_to_onnx_model(*args, **kwargs):
    model_def = make_model(caffe2_net_to_onnx_graph(*args, **kwargs))
    checker.check_model(model_def)
    return model_def


def caffe2_net_to_onnx_graph(predict_net,
                             init_net=None,
                             value_info=None):
    if value_info is None:
        value_info = {}
    if not isinstance(value_info, dict):
        raise ValueError('Please pass value_info as a '
                         'name -> (type, shape) dictionary')
    if init_net:
        initializer = caffe2_init_net_to_initializer(init_net)
        value_info.update({init.name: (init.data_type, init.dims)
                           for init in initializer})
    else:
        initializer = []

    # Check whether we have got type shape info of all input
    missing = (set(list(predict_net.external_input)) -
               set(value_info.keys()))
    if missing:
        raise RuntimeError('Could not find value info of inputs: {}'.format(
            ', '.join(missing)))

    # If there are missing type shape info of output,
    # run the net once to find out!
    missing = (set(list(predict_net.external_output)) -
               set(value_info.keys()))
    if missing:
        inputs = {}
        for name in predict_net.external_input:
            elem_type, shape = value_info[name]
            inputs[name] = np.random.randn(*shape).astype(
                mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type])

        outputs = c2_native_run_net(
            init_net,
            predict_net,
            inputs)

        for name in missing:
            output = outputs[name]
            elem_type = mapping.NP_TYPE_TO_TENSOR_TYPE[output.dtype]
            shape = output.shape
            value_info[name] = (elem_type, shape)

    graph_def = onnx_pb2.GraphProto()
    graph_def.name = predict_net.name
    graph_def.initializer.extend(initializer)
    # This is a mapping from Caffe2 names to ONNX names
    name_map = NameMap()
    graph_def.input.extend(
        onnx.helper.make_tensor_value_info(
            name=name_map.rename(name),
            elem_type=value_info[name][0],
            shape=value_info[name][1])
        for name in predict_net.external_input)
    graph_def.node.extend(
        caffe2_op_to_node_def(op, name_map) for op in predict_net.op)

    all_output = set(sum((list(node.output) for node in graph_def.node),
                         [init.name for init in graph_def.initializer]))
    redundant_output = set(vi.name for vi in graph_def.output) - all_output
    if redundant_output:
        logger.warning(
            'There are graph output not produced by any node or initializer: {}'
            '! Will drop them.'.format(', '.join(redundant_output)))
    graph_def.output.extend(
        onnx.helper.make_tensor_value_info(
            name=name_map.rename(name),
            elem_type=value_info[name][0],
            shape=value_info[name][1])
        for name in predict_net.external_output
        if name in all_output)

    checker.check_graph(graph_def)
    return graph_def
