from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
import psutil


class TestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=0)

    def assertSameOutputs(self, outputs1, outputs2, decimal=7):
        self.assertEqual(len(outputs1), len(outputs2))
        for o1, o2 in zip(outputs1, outputs2):
            np.testing.assert_almost_equal(o1, o2, decimal=decimal)

    @staticmethod
    def report_mem_usage(tag):
        print('Mem usage ({}):'.format(tag),
              psutil.virtual_memory())

    def add_test_case(name, test_func):
        if not name.startswith('test_'):
            raise ValueError('Test name must start with test_: {}'.format(name))
        if hasattr(self, name):
            raise ValueError('Duplicated test name: {}'.format(name))
        setattr(self, name, test_func)
