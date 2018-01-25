set -ex

top_dir=$(dirname $(dirname $(readlink -e "${BASH_SOURCE[0]}")))

# setup ccache
export PATH="/usr/lib/ccache:$PATH"
ccache --max-size 1G
