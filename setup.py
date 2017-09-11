from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from setuptools import setup, find_packages

install_requires = []
if sys.version_info < (3, 4):
    install_requires.append('enum34')

setup(
    name="onnx-caffe2",
    version='0.1.3',
    description="Caffe2 frontend and backend of Open Neural Network Exchange",
    install_requires=install_requires,
    setup_requires=['pytest-runner'],
    tests_require=['numpy', 'pytest-cov'],
    packages=find_packages(),
    author='bddppq',
    author_email='jbai@fb.com',
    url='https://github.com/onnx/onnx-caffe2',
)
