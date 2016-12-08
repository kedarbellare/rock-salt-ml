#!/bin/bash

export PYTHONPATH=$PWD
python3 learn/train.py dummy model -q -start-eps 0.1 -end-eps 0.05 -gamma 0.8 -epochs 2000 -batch-size 50 -i model.h5
