set -ex

export CI=true

export TOP_DIR=$(dirname $(dirname $(readlink -e "${BASH_SOURCE[0]}")))

export OS="$(uname)"

# setup ccache
if [[ "$OS" == "Darwin" ]]; then
    export PATH="/usr/local/opt/ccache/libexec:$PATH"
else
    if [[ -d "/usr/lib/ccache" ]]; then
        export PATH="/usr/lib/ccache:$PATH"
    elif hash ccache > /dev/null; then
        mkdir -p "$TOP_DIR/ccache"
        ln -sf "$(which ccache)" "$TOP_DIR/ccache/cc"
        ln -sf "$(which ccache)" "$TOP_DIR/ccache/c++"
        ln -sf "$(which ccache)" "$TOP_DIR/ccache/gcc"
        ln -sf "$(which ccache)" "$TOP_DIR/ccache/g++"
        ln -sf "$(which ccache)" "$TOP_DIR/ccache/x86_64-linux-gnu-gcc"
        export PATH="$TOP_DIR/ccache:$PATH"
    fi
fi

# setup virtualenv
virtualenv "$TOP_DIR/venv"
source "$TOP_DIR/venv/bin/activate"
pip install -U pip setuptools
