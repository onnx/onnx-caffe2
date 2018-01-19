from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import os
import sys
from setuptools import setup, find_packages, Command
import distutils.command.build
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.install
import subprocess
from textwrap import dedent

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'onnx_caffe2')

################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
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


def build_optimize_onnx():
    optimize_onnx_dir = os.path.join('build', 'optimize-onnx')
    subprocess.check_call(['mkdir', '-p', optimize_onnx_dir])
    subprocess.check_call(['cmake', os.path.join('..', '..', 'optimize-onnx')], cwd=optimize_onnx_dir)
    subprocess.check_call(['make'], cwd=optimize_onnx_dir)

class build(distutils.command.build.build):
    def run(self):
        self.run_command('build_py')
        build_optimize_onnx()

class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        self.run_command('build')
        setuptools.command.develop.develop.run(self)
        source = os.path.join(TOP_DIR, 'build', 'optimize-onnx', 'src', 'optimize-onnx')
        target = os.path.join(sys.prefix, 'bin', '')
        subprocess.check_call(['ln', '-s', '-f', source, target])

class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)
        source = os.path.join(TOP_DIR, 'build', 'optimize-onnx', 'src', 'optimize-onnx')
        target = os.path.join(sys.prefix, 'bin', '')
        subprocess.check_call(['cp', source, target])

cmdclass={
    'create_version': create_version,
    'build_py': build_py,
    'build': build,
    'develop': develop,
    'install': install,
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
