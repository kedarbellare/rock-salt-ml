#!/bin/bash

export PYTHONPATH=$PWD
python3 learn/train.py dummy model -q -start-eps 0.2 -end-eps 0.001 -gamma 0.8 -epochs 10000000 -f axes -i model.h5
