import sys

# HACK: keras prints to stdout which messes with halite
stdout = sys.stdout
stderr = sys.stderr
sys.stdout = open('/dev/null', 'w')
sys.stderr = open('/dev/null', 'w')

from keras import models
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, \
    Convolution1D, Convolution2D, Dropout, Flatten
from keras.utils import np_utils

sys.stdout = stdout
sys.stderr = stderr
