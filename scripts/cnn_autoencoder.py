from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D, Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
from keras.layers import Merge
import numpy as np
import scipy.io as sio

##### Define Parameters
# Path to the input data-file
# FILE_PATH = 'data/raw_sensor_data_sr01_240_rand.mat'
# FILE_PATH = 'data/raw_sensor_data_hosp_240_rand.mat'
# FILE_PATH = 'data/raw_sensor_data_01.mat'
FILE_PATH = 'data/raw_sensor_data_so_02.mat'
# Percentage of data being training data
TRAIN_PERC = 0.9
# Training
LEARNING_RATE = 0.01
DECAY = 0.00  # after 2000 => lr=0.002
DROPOUT_RATE = 0.1
BATCH_SIZE = 200
POOL_LENGTH = 2                 # Best = 4
FILTER_SIZE = 3                 # Best = 3
NUM_HIDDEN_UNITS = 4           # Best = 64
NUM_EPOCHS = 100
BETA = 0.000
QUADR_THETA = False
TRIM_DATA = True
ACT_FUNCTION = 'relu'
LOSS_FUNCTION = 'mse'
TEST_NUM = 'ae_so_small_01'
LOAD_EXISTING_MODEL = False      # Whether existing model should be loaded or new one compiled
UPDATE_MODEL = True             # Whether it is an ae to update
MODEL_NAME = 'cnn_ae_hosp_01'           # Name of existing model


##### Define Functions
# Load the data out of a .mat file
def load_data(file, quadr_theta):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['data_sensor'], float)
    data_y = np.asarray(data_set['data_pose'], float)

    data_x = data_x[:20000]
    data_y = data_y[:20000]

    data_x[data_x == 100.0] = 7.0
    data_x[data_x == -1.0] = 0.0

    # Delete first row of sensor data
    if ('sr' in file) or ('hosp' in file) or TRIM_DATA:
        data_y = data_y[:data_x.shape[0] - 1]
        data_x = data_x[1:]
    
    # Randomly permute data
    perm = np.random.permutation(data_x.shape[0])
    data_x = data_x[perm]
    data_y = data_y[perm]
    
    # Add bias
    # bias = np.ones((data_x.shape[0], 1))
    # data_x = np.append(bias, data_x, 1)
    
    # Split data into train and test data
    train_size = int(len(perm) * TRAIN_PERC)
    train_x = data_x[:train_size, 13:653]
    train_y = data_y[:train_size]
    test_x = data_x[train_size:, 13:653]
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
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    
    return (train_x.shape[1], train_x, train_y, test_x, test_y)


# Create a model with convolutional and maxPool layers
def load_model_ae1(input_dim, data_set_size, l2_reg=0.01, activation='relu',
                   hidden_layer_size=64, hidden_layer_num=3,
                   dropout_rate=0.25):
    print('Building CNN-Model.')
    
    model = Sequential()
    
    pool_length = POOL_LENGTH
    
    # So far:   h = 64/4, 64/2
    #           p = 4
    # layers:   2x1
    
    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation, input_shape=(input_dim, 1)))
    # model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
    #                         activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))
    
    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    # model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
    #                         activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))

    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    # model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
    #                         activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))

    
    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    # model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
    #                         activation=activation))
    model.add(UpSampling1D(length=pool_length))
    
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    # model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
    #                         activation=activation))
    model.add(UpSampling1D(length=pool_length))
    
    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation))
    # model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
    #                         activation=activation))
    model.add(UpSampling1D(length=pool_length))
    
    model.add(Convolution1D(1, 3, border_mode='same'))
    
    model.summary()
    
    return model


# Create a model with convolutional and maxPool layers
def load_model_ae1(input_dim, data_set_size, l2_reg=0.01, activation='relu',
                   hidden_layer_size=64, hidden_layer_num=3,
                   dropout_rate=0.25):
    print('Building CNN-Model.')
    
    model = Sequential()
    
    pool_length = POOL_LENGTH
    
    # So far:   h = 64/4, 64/2
    #           p = 4
    # layers:   2x1
    
    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation, input_shape=(input_dim, 1)))
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))

    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size*2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size*2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))

    # Fully connected Layer
    # model.add(Flatten())
    # model.add(Dense(1280, W_regularizer=l2(l2_reg), init='lecun_uniform'))
    # model.add(BatchNormalization(epsilon=0.001, mode=0))
    # model.add(Activation(activation))
    # model.add(Dropout(0.1))
    #
    # model.add(Dense(16, W_regularizer=l2(l2_reg), init='lecun_uniform'))
    # model.add(BatchNormalization(epsilon=0.001, mode=0))
    # model.add(Activation(activation))
    # model.add(Dropout(0.1))
    #
    # model.add(Dense(1280, W_regularizer=l2(l2_reg), init='lecun_uniform'))
    # model.add(BatchNormalization(epsilon=0.001, mode=0))
    # model.add(Activation(activation))
    # model.add(Dropout(0.1))
    #
    # model.add(Reshape((input_dim/16, hidden_layer_size/2)))
    
    # Deconvolutional Layer 2

    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size*2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size*2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(UpSampling1D(length=pool_length))
    
    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(UpSampling1D(length=pool_length))
  
    model.add(Convolution1D(1, 3, border_mode='same'))
    
    model.summary()
    
    return model


# Create a model with convolutional and maxPool layers
def load_model_ae(input_dim, data_set_size, l2_reg=0.01, activation='relu',
                  hidden_layer_size=64, hidden_layer_num=3,
                  dropout_rate=0.25):
    print('Building CNN-Model.')
    
    model = Sequential()
    
    pool_length = POOL_LENGTH
    
    # So far:   h = 64/4, 64/2
    #           p = 4
    # layers:   2x1
    
    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation, input_shape=(input_dim, 1)))
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))
    
    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))

    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size * 4, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size * 4, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))

    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size * 8, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size * 8, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(MaxPooling1D(pool_length=pool_length, stride=None, border_mode='valid'))

    #########################################################################

    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size * 8, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size * 8, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(UpSampling1D(length=pool_length))

    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size * 4, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size * 4, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(UpSampling1D(length=pool_length))

    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size * 2, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(UpSampling1D(length=pool_length))
    
    # Deconvolutional Layer 2
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(Convolution1D(hidden_layer_size, FILTER_SIZE, border_mode='same',
                            activation=activation))
    model.add(UpSampling1D(length=pool_length))
    
    model.add(Convolution1D(1, 3, border_mode='same'))
    
    model.summary()
    
    return model


if __name__ == '__main__':
    # Load data
    (x_dim, train_x, train_y, test_x, test_y) = load_data(FILE_PATH, QUADR_THETA)
    
    if LOAD_EXISTING_MODEL:
        # Load model from existing data
        # Load json and create model
        json_file = open('data/model_' + MODEL_NAME + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load weights into new model
        model.load_weights('data/model_' + MODEL_NAME + '.h5')
        model.summary()
        print('Loaded model from disk')

    else:
        # Load layers for model
        model = load_model_ae1(x_dim, train_x.shape[0], activation=ACT_FUNCTION,
                              hidden_layer_size=NUM_HIDDEN_UNITS, l2_reg=BETA,
                              dropout_rate=DROPOUT_RATE)
    
    # Optimizers
    # sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    
    # Compile model
    model.compile(loss=LOSS_FUNCTION, metrics=['accuracy'],
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
    print('Loss          = ', LOSS_FUNCTION)
    print('')
    
    history = model.fit(train_x, train_x, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
                        validation_data=(test_x, test_x), shuffle=True)
    
    # Save eval-data
    sio.savemat('data/eval-data_cnn_' + TEST_NUM + '.mat', history.history)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('data/model_cnn_' + TEST_NUM + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('data/model_cnn_' + TEST_NUM + '.h5')
    print('Saved model to disk')

