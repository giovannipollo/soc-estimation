#!/bin/bash
# Check if there is one argument
if [ $# -eq 1 ]; then
    # Check the value of the argument
    if [ $1 -eq 0 ]; then
        python3 src/soc_ocv/train.py
    elif [ $1 -eq 1 ]; then
        python3 src/soc/train.py
    elif [ $1 -eq 2 ]; then
        python3 src/soc/train.py
    fi
else
    echo "Usage: ./run.sh <number>"
fi