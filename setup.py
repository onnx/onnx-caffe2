from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import os
import sys
from setuptools import setup, find_packages, Command
import setuptools.command.build_py
from textwrap import dedent

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'onnx_caffe2')

################################################################################
# Version
################################################################################

VersionInfo = namedtuple('VersionInfo', ['version'])(
    version='0.1.3'
)

################################################################################
# Customized commands
################################################################################

class create_version(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(dedent('''
            version = '{version}'
            '''.format(**dict(VersionInfo._asdict()))))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        setuptools.command.build_py.build_py.run(self)

cmdclass={
    'create_version': create_version,
    'build_py': build_py,
}

################################################################################
# Dependencies
################################################################################

install_requires = []
if sys.version_info < (3, 4):
    install_requires.append('enum34')

################################################################################
# Final
################################################################################

setup(
    name="onnx-caffe2",
    version=VersionInfo.version,
    description="Caffe2 frontend and backend of Open Neural Network Exchange",
    install_requires=install_requires,
    setup_requires=['pytest-runner'],
    tests_require=['numpy', 'pytest-cov'],
    cmdclass=cmdclass,
    packages=find_packages(),
    author='bddppq',
    author_email='jbai@fb.com',
    url='https://github.com/onnx/onnx-caffe2',
)
