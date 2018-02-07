from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import unittest
import onnx.backend.test

import onnx_caffe2.backend as c2

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(c2, __name__)
# Skip vgg to speed up CI, skip others because onnx-caffe2 does not support them yet.
ci_blacklist = (r'(test_vgg19|test_vgg'  # Speed up CI.
                '|test_ceil.*|test_floor.*'  # Does not support Ceil and Floor.
                '|test_hardsigmoid.*|test_pow.*'  # Does not support Hardsigmoid and Pow.
                '|test_mean.*|test_hardmax.*'  # Does not support Mean and Hardmax.
                ')')
if 'CI' in os.environ:
    backend_test.exclude(ci_blacklist)
# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
