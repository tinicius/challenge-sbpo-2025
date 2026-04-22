#!/usr/bin/env bash
# Build the GRASP C++ solver. Keeps the binary next to the sources.
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -DNDEBUG -march=native -pipe \
    -Wall -Wextra -Wno-unused-parameter \
    -o grasp_solver grasp_solver.cpp
