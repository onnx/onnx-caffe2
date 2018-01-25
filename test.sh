#!/bin/bash

set -ex

UNKNOWN=()

# defaults
PARALLEL=0

while [[ $# -gt 0 ]]
do
    arg="$1"
    case $arg in
        -p|--parallel)
            PARALLEL=1
            shift # past argument
            ;;
        *) # unknown option
            UNKNOWN+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done
set -- "${UNKNOWN[@]}" # leave UNKNOWN

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0
]}")
top_dir=$(dirname "$script_path")
TEST_DIR="$top_dir/tests"

pip install pytest-cov nbval psutil tabulate

PYTEST_ARGS="$TEST_DIR"
if [[ $PARALLEL == 1 ]]; then
    pip install pytest-xdist
    PYTEST_ARGS="$PYTEST_ARGS -n 2"
fi

pytest "${PYTEST_ARGS[@]}"
