from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tarfile
import tempfile
import unittest

from subprocess import Popen, PIPE
from six.moves.urllib.request import urlretrieve

import onnx

from tests.test_utils import TestCase

# mostly copied from
# https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py

class TestRoundtrip(TestCase):
    def _roundtrip(self, model_name):
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', '~/.onnx'))
        models_dir = os.getenv('ONNX_MODELS',
                               os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            url = 'https://s3.amazonaws.com/download.onnx/models/{}.tar.gz'.format(
                model_name)

            # On Windows, NamedTemporaryFile can not be opened for a
            # second time
            download_file = tempfile.NamedTemporaryFile(delete=False)
            download_file.close()
            print('Start downloading model {} from {}'.format(
                model_name, url))
            urlretrieve(url, download_file.name)
            print('Done')
            with tarfile.open(download_file.name) as t:
                t.extractall(models_dir)

        pb_path = os.path.join(model_dir, 'model.pb')

        before_roundtrip = onnx.load(pb_path)

        p = Popen(['optimize-onnx'], stdin=PIPE, stdout=PIPE)
        with open(pb_path) as pb:
            out, _ = p.communicate(pb.read())
        assert p.returncode == 0

        after_roundtrip = onnx.load_from_string(out)

        assert onnx.helper.printable_graph(before_roundtrip.graph) \
            == onnx.helper.printable_graph(after_roundtrip.graph)

        with open(pb_path) as pb:
            assert after_roundtrip.SerializeToString() == pb.read()

    # arbitrarily pick one relatively small model to sanity test with
    def test_squeezenet_v3(self):
        self._roundtrip('squeezenet-ir-version-3')

    # testing just to be sure that we no-op instead of breaking on an
    # older IR version.
    def test_squeezenet_v1(self):
        self._roundtrip('squeezenet')

if __name__ == '__main__':
    unittest.main()
