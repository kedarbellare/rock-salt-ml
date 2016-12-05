#!/bin/bash

export PYTHONPATH=$PWD
python3 learn/train.py dummy model -q -start-eps 0.2 -end-eps 0.1 -gamma 0.99 -epochs 1000000 -f axes -i model.h5
