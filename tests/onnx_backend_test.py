from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx.backend.test

import onnx_caffe2.backend as c2

# unittest by default only looks for tests in current scope!
_ = onnx.backend.test.BackendTest(c2).tests


if __name__ == '__main__':
    unittest.main()
