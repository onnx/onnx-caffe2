from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx.backend.test

import onnx_caffe2.backend as c2

# import all test cases at global scope to make them visible to python.unittest
globals().update(onnx.backend.test.BackendTest(c2).test_cases)


if __name__ == '__main__':
    unittest.main()
