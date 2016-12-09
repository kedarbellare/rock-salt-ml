#!/bin/bash

export PYTHONPATH=$PWD
python3 learn/train.py dummy model -q -start-eps 0.4 -end-eps 0.1 -gamma 0.8 -epochs 30000 -batch-size 40 -i model.h5
