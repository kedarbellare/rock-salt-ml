#!/bin/bash

export PYTHONPATH=$PWD
if [ ${1: -4} == ".hlt" ]
then
	python3 learn/train.py $1 model -l -s -f axes
else
	python3 learn/train.py $1 model -l -f axes
fi
