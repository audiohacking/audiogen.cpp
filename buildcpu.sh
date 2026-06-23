#!/bin/sh
set -e
cmake -B build-cpu -DDASHENG_AUDIOGEN_METAL=OFF -DDASHENG_AUDIOGEN_CUDA=OFF
cmake --build build-cpu -j
