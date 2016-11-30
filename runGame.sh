#!/bin/bash

export KERAS_BACKEND=theano
# ./halite -d "30 30" "python3 MyBot.py" "python3 OverkillBot.py"
./halite -d "30 30" "python3 MyBot.py" "python3 RandomBot.py"
