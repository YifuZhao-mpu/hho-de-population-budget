#!/usr/bin/env bash
# Builds the official CEC2014, CEC2017 and CEC2022 benchmark suites as shared
# libraries.
# Sources are the competition C/C++ code (P. N. Suganthan's official
# distributions) with five mechanical edits (headers, globals, a settable
# input-data directory, %Lf -> %lf in the matrix and shift loaders, C linkage);
# no objective expression is changed. See README.md, "Benchmark provenance".
set -e
cd "$(dirname "$0")"
g++ -O3 -fPIC -w -shared -o libcec14.so patched_cec14_test_func.cpp -lm
g++ -O3 -fPIC -w -shared -o libcec17.so patched_cec17_test_func.cpp -lm
g++ -O3 -fPIC -w -shared -o libcec22.so patched_cec22_test_func.cpp -lm
echo "built: libcec14.so libcec17.so libcec22.so"
