#!/bin/bash

export PYTHONPATH=$PWD
if [ ${1: -4} == ".hlt" ]
then
	python3 learn/train.py $1 model -s
else
	python3 learn/train.py $1 model -epochs 5 -checkpoint-samples 1000000 -w 12
fi
