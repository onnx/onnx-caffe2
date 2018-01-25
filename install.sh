#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname "$script_path")
REPOS_DIR="$top_dir/third_party"
BUILD_DIR="$top_dir/build"
mkdir -p "$BUILD_DIR"

_pip_install() {
    if [[ -n "$CI" ]]; then
        ccache -z
    fi
    if [[ -n "$CI" ]]; then
        time pip install "$@"
    else
        pip install "$@"
    fi
    if [[ -n "$CI" ]]; then
        ccache -s
    fi
}

# Install caffe2
_pip_install -b "$BUILD_DIR/caffe2" "file://$REPOS_DIR/caffe2#egg=caffe2"
python -c 'from caffe2.python import build; from pprint import pprint; pprint(build.build_options)'

# Install onnx
_pip_install -b "$BUILD_DIR/onnx" "file://$REPOS_DIR/onnx#egg=onnx"

# Install onnx-caffe2
_pip_install .
