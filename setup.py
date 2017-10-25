from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import os
import sys
from setuptools import setup, find_packages, Command
import setuptools.command.build_py
import setuptools.command.develop
import subprocess
from textwrap import dedent

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'onnx_caffe2')

################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except subprocess.CalledProcessError:
    git_version = None

VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
    version='0.2.1',
    git_version=git_version
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
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        setuptools.command.develop.develop.run(self)

cmdclass={
    'create_version': create_version,
    'build_py': build_py,
    'develop': develop,
}

################################################################################
# Dependencies
################################################################################

install_requires = ['click']
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
    tests_require=['numpy', 'pytest-cov', 'psutil'],
    cmdclass=cmdclass,
    packages=find_packages(),
    author='bddppq',
    author_email='jbai@fb.com',
    url='https://github.com/onnx/onnx-caffe2',
    entry_points={
        'console_scripts': [
            'convert-caffe2-to-onnx = onnx_caffe2.bin.conversion:caffe2_to_onnx',
            'convert-onnx-to-caffe2 = onnx_caffe2.bin.conversion:onnx_to_caffe2'
        ]
    },
)
