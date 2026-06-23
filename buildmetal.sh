#!/bin/sh
set -e
cmake -B build-metal -DDASHENG_AUDIOGEN_METAL=ON
cmake --build build-metal -j
