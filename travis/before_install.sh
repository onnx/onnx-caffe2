#!/bin/bash

source "$(dirname $(readlink -e "${BASH_SOURCE[0]}"))/setup.sh"

# Install GCC 5
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y --no-install-recommends g++-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 \
     --slave /usr/bin/g++ g++ /usr/bin/g++-5

# Install protobuf
pb_version="2.6.1"
pb_dir="~/.cache/pb"
mkdir -p "$pb_dir"
wget -qO- "https://github.com/google/protobuf/releases/download/v$pb_version/protobuf-$pb_version.tar.gz" | tar -xz -C "$pb_dir" --strip-components 1
ccache -z
cd "$pb_dir" && ./configure && make && make check && sudo make install && sudo ldconfig
ccache -s

# Update all existing python packages
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
