#!/bin/bash

export PYTHONPATH=$PWD
python3 learn/train.py $1 model -l -s -f axes
