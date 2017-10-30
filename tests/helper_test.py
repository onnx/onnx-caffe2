from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from onnx_caffe2.helper import dummy_name

from tests.test_utils import TestCase


class TestCaffe2Basic(TestCase):
    def test_dummy_name(self):
        names_1 = [dummy_name() for _ in range(3)]
        dummy_name([])
        names_2 = [dummy_name() for _ in range(3)]
        self.assertEqual(names_1, names_2)

        dummy_name(names_1)
        names_3 = [dummy_name() for _ in range(3)]
        self.assertFalse(set(names_1) & set(names_3))


if __name__ == '__main__':
    unittest.main()
