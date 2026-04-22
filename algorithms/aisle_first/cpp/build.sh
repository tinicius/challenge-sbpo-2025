#!/usr/bin/env bash
# Build the aisle-GRASP C++ solver. Keeps the binary next to the sources.
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -DNDEBUG -march=native -pipe \
    -Wall -Wextra -Wno-unused-parameter \
    -o aisle_grasp_solver aisle_grasp_solver.cpp
