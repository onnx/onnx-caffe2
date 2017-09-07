from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from setuptools import setup, find_packages

setup(
    name="onnx-caffe2",
    version='0.1.1',
    description="Caffe2 frontend and backend of Open Neural Network Exchange",
    install_requires=['enum'],
    setup_requires=['pytest-runner'],
    tests_require=['numpy', 'pytest-cov'],
    packages=find_packages(),
    author='bddppq',
    author_email='jbai@fb.com',
    url='https://github.com/onnx/onnx-caffe2',
)
