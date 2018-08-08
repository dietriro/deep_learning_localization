from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2

import numpy as np
import scipy.io as sio


MODEL_NAME = 'cnn_240_so_larged_0'
FILE = 'data/robot_240_so_driving_acml.mat'
SAVE_FILE = 'data/robot_240_so_driving_acml_multi.mat'

### LOAD DATA

# Read data from .mat file
data_set = sio.loadmat(FILE)
data_x = np.asarray(data_set['scan'], float)
data_y = np.asarray(data_set['odom'], float)
data_amcl = np.asarray(data_set['acml'], float)

data_x[data_x > 7.0] = 7.0
data_x[data_x < -0.1] = 0.0

# Reshape data
# if 'cnn' in MODEL_TYPE:
data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], 1))
  
  
### LOAD MODEL
pred = np.zeros(data_y.shape)
for i in range(1, 8):

    # Load model from existing data
    # Load json and create model
    json_file = open('data/model_' + MODEL_NAME + str(i) + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into new model
    model.load_weights('data/model_' + MODEL_NAME + str(i) +  '.h5')
    print('Loaded model from disk')


    ### PREDICT VALUES
    adam = Adam(lr=0.006, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_absolute_error', #metrics=['accuracy'],
                          optimizer=adam)
    
    
    
    # for i in range(data_x.shape[0]):
    pred += model.predict(data_x, batch_size=1, verbose=0)

pred /= 7

### SAVE VALUES TO MAT
sio.savemat(SAVE_FILE, {'single_pred': pred, 'odom': data_y, 'amcl': data_amcl, 'scan':data_x})


