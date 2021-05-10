#!/bin/bash

for (( i = 0; i < 10; i++)); do
    ./matmul ./data/input_4096.txt ./data/output_4096.txt 0
done