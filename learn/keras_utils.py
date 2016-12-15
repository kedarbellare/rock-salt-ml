import os
import sys

# HACK: keras prints to stdout which messes with halite
stdout = sys.stdout
stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

from keras import models
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Activation, \
    Convolution1D, Convolution2D, Dropout, Flatten
from keras.optimizers import SGD, RMSprop, Nadam, Adam
from keras.utils import np_utils

sys.stdout = stdout
sys.stderr = stderr
