#!/bin/bash

set -ex

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0
]}")
top_dir=$(dirname "$script_path")
TEST_DIR="$top_dir/tests"

pip install pytest-cov nbval psutil tabulate
pytest "$TEST_DIR"
