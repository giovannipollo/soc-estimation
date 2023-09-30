#!/bin/bash
# Check if there is one argument
if [ $# -eq 1 ]; then
    # Check the value of the argument
    if [ $1 -eq 0 ]; then
        python3 src/soc_ocv/train.py
    elif [ $1 -eq 1 ]; then
        python3 src/soc/train.py
    elif [ $1 -eq 2 ]; then
        echo "Cleaning images in plots subdirectories..."
        rm plots/physics/*.png
        rm plots/test/*.png
        rm plots/train/*.png
        echo "Cleaning the log files..."
        rm -rf log/*.log
        echo "Starting the train..."
        python3 src/soc/train.py
    fi
else
    echo "Usage: ./run.sh <number>"
fi