#!/bin/bash

set -ex

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common";

[[ -n "$CAFFE2_VERSION" ]] || die "CAFFE2_VERSION not set"
[[ -n "$PYTHON_VERSION" ]] || die "PYTHON_VERSION not set"

if [[ $PYTHON_VERSION == 2.* ]]; then
    rm -rf venv2 && virtualenv venv2
    source venv2/bin/activate
elif [[ $PYTHON_VERSION == 3.* ]]; then
    rm -rf venv3 && python3 -m venv venv3
    source venv3/bin/activate
else
    die "Unknown Python version: $PYTHON_VERSION"
fi

pip install -U setuptools pip

deps_dir="$workdir/deps"

# setup caffe2
c2_dir="$deps_dir/caffe2"
c2_install_dir="$build_cache_dir/caffe2/$CAFFE2_VERSION"

# install
export PYTHONPATH=$c2_install_dir
export LD_LIBRARY_PATH="$c2_install_dir/lib"
pip install numpy future
if ! python -c 'import caffe2'; then
    rm -rf $c2_dir $c2_install_dir
    git clone https://github.com/caffe2/caffe2.git $c2_dir && cd $c2_dir
    git checkout "$CAFFE2_VERSION" && git submodule update --init

    ccache -z
    mkdir build && cd build
    cmake \
        $(python $c2_dir/scripts/get_python_cmake_flags.py) \
        -DCMAKE_INSTALL_PREFIX:PATH="$c2_install_dir" \
        ..
    VERBOSE=1 make -j16
    make install
    ccache -s
fi

onnx_c2_dir="$workdir/src"
pip install -e $onnx_c2_dir

# install onnx
onnx_dir="$deps_dir/onnx"
rm -rf $onnx_dir && git clone "https://github.com/onnx/onnx.git" "$onnx_dir" --recursive
pip install "$onnx_dir"
pip install pytest-cov psutil pytest-xdist

# run caffe2 tests
cd "$onnx_c2_dir"
pytest -s -n 2
