from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
import numpy as np
import scipy.io as sio
from math import pi

##### Define Parameters
# Path to the input data-file
FILE_PATH = 'data/raw_sensor_data_01.mat'
# FILE_PATH = 'data/raw_sensor_data_360_rand.mat'
# Percentage of data being training data
TRAIN_PERC = 0.9
# Training
LEARNING_RATE = 0.01
DECAY = 0.005  # after 2000 => lr=0.002
DROPOUT_RATE = 0.1
BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 4
NUM_EPOCHS = 400
BETA = 0.000
QUADR_THETA = False
ACT_FUNCTION = 'sigmoid'
TEST_NUM = '03'


##### Define Functions
# Convert data from angle/range to x/y image
def data_to_euclid(data_in, save=False):
    # Input data
    min = -np.deg2rad(120)  # rad
    max = np.deg2rad(120)  # rad
    
    # Image/Map
    img_size = 384
    
    # Conversion parameters
    sin_values = np.sin(np.linspace(min, max, data_in.shape[1]))
    cos_values = np.cos(np.linspace(min, max, data_in.shape[1]))
    new_data_x = np.zeros((data_in.shape[0], img_size, img_size))

    max_value = np.max(data_in)+0.1

    # Converting values into map
    for i in range(data_in.shape[0]):
        x = sin_values * data_in[i, :]
        y = cos_values * data_in[i, :]

        # new_data_x[i] = points_to_map(x[data_in[i]<7], y[data_in[i]<7], img_size)
        new_data_x[i] = points_to_map(x, y, img_size, -max_value, max_value)

    sio.savemat('data/sensor_to_map', mdict={'raw_data': new_data_x[90-110]})

    return new_data_x
 
        
def points_to_map(x, y, size, min_value, max_value):
    
    factor = size/(2*max_value)
    x = np.rint(x*factor+size/2).astype(int)
    y = np.rint(y*factor+size/2).astype(int)

    map = np.zeros((size, size))
    map[x, y] = 1
    
    return map


# Load the data out of a .mat file
def load_data(file, quadr_theta):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['data_sensor'], float)
    data_y = np.asarray(data_set['data_pose'], float)

    data_x[data_x > 7.0] = 7.0
    data_x[data_x < 0] = 0.0
    
    data_y = data_y[:data_y.shape[0]-1]
    
    print(data_x.shape)
    print(data_y.shape)

    # [data_x, data_y] = data_to_euclid(data_x, data_y)
    
    # Randomly permute data
    # perm = np.random.permutation(data_x.shape[0])
    # data_x = data_x[perm]
    # data_y = data_y[perm]
    
    # Add bias
    # bias = np.ones((data_x.shape[0], 1))
    # data_x = np.append(bias, data_x, 1)
    
    # Split data into train and test data
    train_size = int(data_x.shape[0] * TRAIN_PERC)
    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    test_x = data_x[train_size:]
    test_y = data_y[train_size:]
    
    # Update theta
    if quadr_theta:
        train_y[:, 3:4] = train_y[:, 3:4] * train_y[:, 3:4]
        test_y[:, 3:4] = test_y[:, 3:4] * test_y[:, 3:4]
    
    # Define arrays as float32
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('float32')
    
    # Reshape data
    # train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    # test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    
    return (train_x.shape[1], train_x, train_y, test_x, test_y)


# Create a model with convolutional and maxPool layers
def load_model_cnn_working(input_dim, data_set_size, l2_reg=0.01, activation='relu',
                   hidden_layer_size=64, hidden_layer_num=3,
                   dropout_rate=0.25):
    print('Building CNN-Model.')
    
    model = Sequential()

    # Convolutional Layer 1
    model.add(Convolution2D(hidden_layer_size, 3, 3, border_mode='same',
                            input_shape=(1, input_dim, input_dim)))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(8, 8), border_mode='valid'))
    model.add(Dropout(dropout_rate))

    # Convolutional Layer 2
    model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(8, 8), border_mode='valid'))
    model.add(Dropout(dropout_rate))
    
    # # Convolutional Layer 3
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(MaxPooling2D(pool_size=(4, 4), border_mode='valid'))
    # model.add(Dropout(dropout_rate))
    
    # # Convolutional Layer 4
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    # model.add(Dropout(dropout_rate))
    
    
    # Fully connected Layer
    model.add(Flatten())

    # model.add(Dense(hidden_layer_size * 8, W_regularizer=l2(l2_reg),
    #                 init='lecun_uniform'))
    # model.add(BatchNormalization(epsilon=0.001, mode=0))
    # model.add(Activation(activation))
    # model.add(Dropout(dropout_rate / 2))

    model.add(Dense(hidden_layer_size * 8, W_regularizer=l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(hidden_layer_size * 8, W_regularizer=l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(hidden_layer_size, W_regularizer=l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(3))
    model.add(Activation("linear"))

    model.summary()

    return model


# Create a model with convolutional and maxPool layers
def load_model_cnn(input_dim, data_set_size, l2_reg=0.01, activation='relu',
                   hidden_layer_size=64, hidden_layer_num=3,
                   dropout_rate=0.25):
    print('Building CNN-Model.')
    
    model = Sequential()
    
    # Convolutional Layer 1
    model.add(Convolution2D(hidden_layer_size, 3, 3, border_mode='same',
                            input_shape=(1, input_dim, input_dim)))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(dropout_rate))
    
    # Convolutional Layer 2
    model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(dropout_rate))
    
    # Convolutional Layer 2
    model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(dropout_rate))
    
    # Convolutional Layer 2
    model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(dropout_rate))
    
    # Convolutional Layer 2
    model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(dropout_rate))
    
    # Convolutional Layer 2
    model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(hidden_layer_size * 2, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(dropout_rate))
    
    # # Convolutional Layer 3
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(MaxPooling2D(pool_size=(4, 4), border_mode='valid'))
    # model.add(Dropout(dropout_rate))
    
    # # Convolutional Layer 4
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(Convolution2D(hidden_layer_size * 4, 3, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    # model.add(Dropout(dropout_rate))
    
    
    # Fully connected Layer
    model.add(Flatten())
    
    # model.add(Dense(hidden_layer_size * 8, W_regularizer=l2(l2_reg),
    #                 init='lecun_uniform'))
    # model.add(BatchNormalization(epsilon=0.001, mode=0))
    # model.add(Activation(activation))
    # model.add(Dropout(dropout_rate / 2))
    
    model.add(Dense(512, W_regularizer=l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate / 2))
    
    model.add(Dense(512, W_regularizer=l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate / 2))
    
    model.add(Dense(hidden_layer_size * 16, W_regularizer=l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate / 2))
    
    model.add(Dense(3))
    model.add(Activation("linear"))
    
    model.summary()
    
    return model


if __name__ == '__main__':
    # Load data
    (x_dim, train_x, train_y, test_x, test_y) = load_data(FILE_PATH, QUADR_THETA)
    
    train_x = data_to_euclid(train_x, save=True)
    test_x = data_to_euclid(test_x)
    
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1], train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1], test_x.shape[1]))
    
    x_dim = train_x.shape[2]
    
    # Load layers for model
    model = load_model_cnn(x_dim, train_x.shape[0], activation=ACT_FUNCTION,
                           hidden_layer_size=NUM_HIDDEN_UNITS, l2_reg=BETA,
                           dropout_rate=DROPOUT_RATE)
    
    # Optimizers
    # sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    
    # Compile model
    model.compile(loss='mean_absolute_error',  # metrics=['accuracy'],
                  optimizer=adam)
    
    print('Start training CNN.')
    
    # Print parameter information
    print('')
    print('Parameters:')
    
    print('Learning Rate = ', LEARNING_RATE)
    print('Dropout       = ', DROPOUT_RATE)
    print('Batch Size    = ', BATCH_SIZE)
    print('Hidden Layer  = ', NUM_HIDDEN_UNITS)
    print('Optimizer     =  Adam')
    print('Loss          =  mean_absolute_error')
    print('')
    
    history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
                        validation_data=(test_x, test_y), shuffle=True)
    
    # Save eval-data
    sio.savemat('data/eval-data_cnn_2d_' + TEST_NUM + '.mat', history.history)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('data/model_cnn_2d_' + TEST_NUM + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('data/model_cnn_2d_' + TEST_NUM + '.h5')
    print('Saved model to disk')

