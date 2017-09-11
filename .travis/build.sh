#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common";

[[ -n "$CAFFE2_VERSION" ]] || die "CAFFE2_VERSION not set"

onnx_c2_dir="$PWD"
pip install $onnx_c2_dir

# setup caffe2
c2_dir="$workdir/caffe2"
c2_install_dir="$build_cache_dir/caffe2/$CAFFE2_VERSION"


if [[ $TRAVIS_PYTHON_VERSION == 2.* ]]; then
    libpython=$(python-config --prefix)/lib/libpython"$TRAVIS_PYTHON_VERSION".so
elif [[ $TRAVIS_PYTHON_VERSION == 3.* ]]; then
    libpython=$(python-config --prefix)/lib/libpython"$TRAVIS_PYTHON_VERSION"m.so
else
    die "Unknown Python version: $TRAVIS_PYTHON_VERSION"
fi
[[ -e "$libpython" ]] || die "Could not find libpython"

# install
export PYTHONPATH=$c2_install_dir
export LD_LIBRARY_PATH="$c2_install_dir/lib"
pip install numpy future

if ! python -c 'import caffe2'; then
    rm -rf $c2_install_dir
    git clone https://github.com/caffe2/caffe2.git $c2_dir && cd $c2_dir
    git checkout "$CAFFE2_VERSION" && git submodule update --init

    ccache -z
    mkdir build && cd build
    cmake \
        -DPYTHON_INCLUDE_DIR="$(python -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())')" \
        -DPYTHON_LIBRARY="$libpython" \
        -DCMAKE_INSTALL_PREFIX:PATH="$c2_install_dir" \
        ..
    VERBOSE=1 make -j16
    make install
    ccache -s
fi

# install onnx
onnx_dir="$workdir/onnx"
git clone "https://github.com/onnx/onnx.git" "$onnx_dir" --recursive
pip install "$onnx_dir"
pip install pytest-cov

# run caffe2 tests
cd "$workdir"
pytest "$onnx_c2_dir/tests/caffe2_ref_test.py"
