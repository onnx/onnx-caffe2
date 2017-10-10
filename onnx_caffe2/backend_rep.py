from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from onnx.backend.base import BackendRep, namedtupledict

class Caffe2Rep(BackendRep):
    def __init__(self, predict_net, workspace, uninitialized):
        super(Caffe2Rep, self).__init__()
        self.predict_net = predict_net
        self.workspace = workspace
        # The list of uninitialized external_inputs in workspace, we need this to
        # pair the name with given sequence inputs.
        self.uninitialized = uninitialized
        self.net_created = False

    @property
    def _name_scope(self):
        if self.predict_net.device_option.device_type == caffe2_pb2.CUDA:
            return 'gpu_{}'.format(self.predict_net.device_option.cuda_gpu_id)
        return ''

    def run(self, inputs, **kwargs):
        super(Caffe2Rep, self).run(inputs, **kwargs)
        with self.workspace:
            with core.DeviceScope(self.predict_net.device_option):
                if isinstance(inputs, dict):
                    with core.NameScope(self._name_scope):
                        for key, value in inputs.items():
                            workspace.FeedBlob(key, value)
                elif isinstance(inputs, list) or isinstance(inputs, tuple):
                    if len(self.uninitialized) != len(inputs):
                        raise RuntimeError('Expected {} values for uninitialized '
                                           'graph inputs ({}), but got {}.'.format(
                                               len(self.uninitialized),
                                               ', '.join(self.uninitialized),
                                               len(inputs)))
                    for i, value in enumerate(inputs):
                        # namescope already baked into protobuf
                        workspace.FeedBlob(self.uninitialized[i], value)
                else:
                    # single input
                    workspace.FeedBlob(self.uninitialized[0], inputs)
                if not self.net_created:
                    workspace.CreateNet(self.predict_net)
                    self.net_created = True
                workspace.RunNet(self.predict_net.name)
            output_values = [workspace.FetchBlob(name)
                             for name in self.predict_net.external_output]
            return namedtupledict('Outputs',
                                  self.predict_net.external_output)(*output_values)
