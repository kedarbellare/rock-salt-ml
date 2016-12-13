#!/bin/bash

export PYTHONPATH=$PWD
python3 learn/train.py dummy model -q -start-eps 0.2 -end-eps 0.1 -gamma 0.9 -epochs 200000 -batch-size 32 -i model.h5
