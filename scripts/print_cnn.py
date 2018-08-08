from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
import theano
import numpy as np
import scipy.io as sio




model = Sequential()

model.add(Convolution1D(16, 3, border_mode='same', input_shape=(667, 1)))
# model.add(Activation('relu'))

sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])

import keras.backend as K
f = K.function([K.learning_phase(), model.layers[0].input], [model.layers[4].output])

