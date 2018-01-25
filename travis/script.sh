#!/bin/bash

source "$(dirname $(readlink -e "${BASH_SOURCE[0]}"))/setup.sh"

time "$top_dir/test.sh"
