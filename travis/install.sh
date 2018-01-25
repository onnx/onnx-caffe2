#!/bin/bash

source "$(dirname $(readlink -e "${BASH_SOURCE[0]}"))/setup.sh"

time CMAKE_ARGS='-DUSE_ATEN=ON -DUSE_OPENMP=ON' "$top_dir/install.sh"
