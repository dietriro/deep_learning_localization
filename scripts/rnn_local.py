from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
import numpy as np
import scipy.io as sio

##### Define Parameters
# Path to the input data-file
# FILE_PATH = 'data/raw_sensor_data_01.mat'
FILE_PATH = 'data/robot_240_driving.mat'
# Percentage of data being training data
TRAIN_PERC = 0.9
# Training
LEARNING_RATE = 0.01
DECAY = 0.005  # after 6000 => lr=0.0016
DROPOUT_RATE = 0.25
BATCH_SIZE = 200  # 400 > 200 > 100 -> in the end
NUM_HIDDEN_UNITS = 64
NUM_EPOCHS = 50
BETA = 0.000
QUADR_THETA = False
ACT_FUNCTION = 'sigmoid'
TEST_NUM = '08'  # Defines the name of the saved file
LOAD_EXISTING_MODEL = True  # Whether existing model should be loaded or new one compiled
MODEL_NAME = 'cnn_07'  # Name of existing model


##### Define Functions
# Load the data out of a .mat file
def load_data(file, quadr_theta):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['laserscan'], float)
    data_y = np.asarray(data_set['odometry'], float)
    
    data_x[data_x == np.inf] = 7.0
    data_x[data_x == -np.inf] = 0.0
    
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
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    
    return (train_x.shape[1], train_x, train_y, test_x, test_y)


def add_rnn_to_model(model, hidden_layer_size=64, input_dim=667, activation='sigmoid',
                     dropout_rate=0.1, seq_len=128, time=2000):
    

    model_new = Sequential()

    # Convolutional Layer 1
    model_new.add(TimeDistributed(Convolution1D(hidden_layer_size, 3,
                                                border_mode='same',
                                                weights=model.layers[0].get_weights()),
                                  batch_input_shape=(1, None, input_dim, 1)))
    model_new.add(Activation(activation))
    model_new.add(TimeDistributed(Convolution1D(hidden_layer_size, 3,
                                                border_mode='same',
                                                weights=model.layers[2].get_weights())))
    model_new.add(Activation(activation))
    model_new.add(TimeDistributed(MaxPooling1D(pool_length=8, stride=None,
                                               border_mode='valid')))
    model_new.add(Dropout(dropout_rate))

    # Convolutional Layer 2
    model_new.add(TimeDistributed(Convolution1D(hidden_layer_size * 2, 3,
                                                border_mode='same',
                                                weights=model.layers[6].get_weights())))
    model_new.add(Activation(activation))
    model_new.add(TimeDistributed(Convolution1D(hidden_layer_size * 2, 3,
                                                border_mode='same',
                                                weights=model.layers[8].get_weights())))
    model_new.add(Activation(activation))
    model_new.add(TimeDistributed(MaxPooling1D(pool_length=8, stride=None,
                                               border_mode='valid')))
    model_new.add(Dropout(dropout_rate))

    # Fully connected Layer
    model_new.add(TimeDistributed(Flatten()))
    # model_new.add(TimeDistributed(Reshape((2000, 1280))))
    
    # Recurrent Layer 1
    model_new.add(LSTM(128, return_sequences=True, consume_less='cpu'))
    model_new.add(LSTM(128, return_sequences=True, consume_less='cpu'))
    
    model_new.add(Dense(3))
    model_new.add(Activation("linear"))
    
    model_new.summary()

    return model

if __name__ == '__main__':
    
    # Load data
    (x_dim, train_x, train_y, test_x, test_y) = load_data(FILE_PATH, QUADR_THETA)
    
    # Load model from existing data
    # Load json and create model
    json_file = open('data/model_' + MODEL_NAME + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into new model
    model.load_weights('data/model_' + MODEL_NAME + '.h5')
    print('Loaded model from disk')
    
    # Add RNN to the network
    model = add_rnn_to_model(model)
    
    # Optimizers
    # sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    # adag = Adagrad(lr=LEARNING_RATE, epsilon=1e-08, decay=0.0)      # Not bad with lr=0.01
    # adad = Adadelta(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
    
    # Compile model
    model.compile(loss='mean_absolute_error', metrics=['accuracy'],
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
    sio.savemat('data/eval-data_crnn_' + TEST_NUM + '.mat', history.history)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('data/model_crnn_' + TEST_NUM + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('data/model_crnn_' + TEST_NUM + '.h5')
    print('Saved model to disk')

