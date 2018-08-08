from __future__ import print_function
from keras.models import Sequential, model_from_json, Model
from keras.layers import Input, Merge, Reshape
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2, activity_l2
import keras
import numpy as np
import scipy.io as sio

##### Define Parameters
# Path to the input data-file
FILE_PATH = 'data/robot_240_driving.mat'
# FILE_PATH = 'data/raw_sensor_data_360_rand.mat'
# Percentage of data being training data
TRAIN_PERC = 0.894
# Training
LEARNING_RATE = 0.01
DECAY = 0.00                   # after 6000 => lr=0.0016
DROPOUT_RATE = 0.1
BATCH_SIZE = 5                # 400 > 200 > 100 -> in the end
NUM_HIDDEN_UNITS = 16
NUM_EPOCHS = 2000
NUM_EXAMPLES = 126              # 153/126/119/102/63/51 -> 14/17/18/21/34/42
BETA = 0.000
QUADR_THETA = False
ACT_FUNCTION = 'sigmoid'
TEST_NUM = '_240_0'         # Defines the name of the saved file
LOAD_EXISTING_MODEL = False     # Whether existing model should be loaded or new one compiled
MODEL_NAME = 'cnn_ae_01'           # Name of existing model


##### Define Functions
# Load the data out of a .mat file
def load_data_cnn(file, quadr_theta):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['data_sensor'], float)
    data_y = np.asarray(data_set['data_pose'], float)
    
    data_x[data_x == 100.0] = 7.0
    data_x[data_x == -1.0] = 0.0
    
    # Randomly permute data
    perm = np.random.permutation(data_x.shape[0])
    data_x = data_x[perm]
    data_y = data_y[perm]
    
    # Add bias
    # bias = np.ones((data_x.shape[0], 1))
    # data_x = np.append(bias, data_x, 1)
    
    # Split data into train and test data
    train_size = int(len(perm) * TRAIN_PERC)
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

def load_data_rnn(file):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['laserscan'], float)
    data_z = np.asarray(data_set['cmd_vel'], float)
    data_y = np.asarray(data_set['odometry'], float)
    
    # Shift data
    data_x = data_x[1:]
    data_z = data_z[:data_z.shape[0] - 1]
    data_y = data_y[1:]
    
    data_x[data_x == np.inf] = 7.0
    data_x[data_x == -np.inf] = 0.0
    
    # Split data into train and test data
    train_size = int(data_x.shape[0] * TRAIN_PERC)
    print(data_x.shape[0])
    print(train_size)
    train_x = data_x[:train_size]
    train_z = data_z[:train_size]
    train_y = data_y[:train_size]
    test_x = data_x[train_size:]
    test_z = data_z[train_size:]
    test_y = data_y[train_size:]
    
    # Reshape data
    ex_size = int(train_size/NUM_EXAMPLES)
    train_x = train_x.reshape((NUM_EXAMPLES, ex_size, train_x.shape[1], 1))
    train_z = train_z.reshape((NUM_EXAMPLES, ex_size, train_z.shape[1]))
    train_y = train_y.reshape((NUM_EXAMPLES, ex_size, train_y.shape[1]))
    
    
    # Define arrays as float32
    train_x = train_x.astype('float32')
    train_z = train_z.astype('float32')
    train_y = train_y.astype('float32')
    test_x = test_x.astype('float32')
    test_z = test_z.astype('float32')
    test_y = test_y.astype('float32')
    
    # Reshape data
    # train_x = train_x.reshape((1, train_x.shape[0], train_x.shape[1], 1))
    # train_z = train_z.reshape((1, train_z.shape[0], train_z.shape[1]))
    # train_y = train_y.reshape((1, train_y.shape[0], train_y.shape[1]))
    test_x = test_x.reshape((1, test_x.shape[0], test_x.shape[1], 1))
    test_z = test_z.reshape((1, test_z.shape[0], test_z.shape[1]))
    test_y = test_y.reshape((1, test_y.shape[0], test_y.shape[1]))

    return (train_x.shape[2], train_x, train_z, train_y, test_x, test_z, test_y)


def update_model_to_rnn(cnn_model, sensor_input_dim, state_input_dim=5,
                        activation='relu', hidden_layer_size=64, dropout_rate=0.25):

    # sensor_input = Input(shape=(None, 1, 1, sensor_input_dim))
    sensor_input = Input(shape=(None, sensor_input_dim, 1))

    x = TimeDistributed(Convolution1D(hidden_layer_size, 3, border_mode='same',
                        activation=activation))(sensor_input)
    x = TimeDistributed(Convolution1D(hidden_layer_size, 3, border_mode='same',
                      activation=activation))(x)
    x = TimeDistributed(MaxPooling1D(pool_length=8, stride=None, border_mode='valid'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    
    x = TimeDistributed(Convolution1D(hidden_layer_size, 3, border_mode='same',
                      activation=activation))(x)
    x = TimeDistributed(Convolution1D(hidden_layer_size, 3, border_mode='same',
                      activation=activation))(x)
    x = TimeDistributed(MaxPooling1D(pool_length=8, stride=None, border_mode='valid'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    
    encoded_scan = TimeDistributed(Flatten())(x)
    
    state_input = Input(shape=(state_input_dim, 1))
    
    x = keras.layers.concatenate([encoded_scan, state_input])
    
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    
    x = TimeDistributed(Dense(3))(x)
    
    pred = Activation('linear')(x)
    
    return Model([sensor_input, state_input], pred)


def update_model_to_rnn_seq(cnn_model, sensor_input_dim, state_input_dim=2,
                        activation='relu', hidden_layer_size=64, dropout_rate=0.25):
    
    # Pop all unnecessary layers
    # for i in range(11):
    #     layer = cnn_model.layers.pop()
    
    sensor_model = Sequential()
    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size, 3,
                                                   border_mode='same',
                                                   activation='relu'),
                                     input_shape=(None, sensor_input_dim, 1)))
    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size, 3, border_mode='same',
                                                   activation='relu')))
    sensor_model.add(TimeDistributed(MaxPooling1D(pool_length=8, stride=None, border_mode='valid')))
    # sensor_model.add(TimeDistributed(Dropout(dropout_rate)))
    
    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size*2, 3, border_mode='same',
                                      activation='relu')))
    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size*2, 3, border_mode='same',
                                      activation='relu')))
    sensor_model.add(TimeDistributed(MaxPooling1D(pool_length=8, stride=None, border_mode='valid')))
    # sensor_model.add(TimeDistributed(Dropout(dropout_rate)))
    sensor_model.add(TimeDistributed(Flatten()))
    sensor_model.add(TimeDistributed(Dense(hidden_layer_size, activation=activation)))
    sensor_model.add(TimeDistributed(Dense(hidden_layer_size, activation=activation)))
    sensor_model.add(TimeDistributed(Dense(3, activation=activation)))

    state_input = Input(shape=(None, state_input_dim))
    state_model = Model(state_input, state_input)
    
    driving_model = Sequential()
    driving_model.add(Merge([sensor_model, state_model], mode='concat'))
    driving_model.add(LSTM(128, return_sequences=True))
    driving_model.add(LSTM(128, return_sequences=True))
    driving_model.add(TimeDistributed(Dense(3, activation='linear')))
    driving_model.summary()
    
    return driving_model


def update_model_to_rnn_sensor_only(cnn_model, sensor_input_dim, state_input_dim=2,
                                    activation='relu', hidden_layer_size=64,
                                    dropout_rate=0.25):
    # # Pop all unnecessary layers
    # for i in range(11):
    #     layer = cnn_model.layers.pop()

    sensor_model = Sequential()
    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size, 3,
                                                   border_mode='same',
                                                   activation='relu',
                                                   weights=cnn_model.layers[0].get_weights(),
                                                   trainable=False),
                                     input_shape=(None, sensor_input_dim, 1)))
    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size, 3,
                                                   border_mode='same',
                                                   activation='relu',
                                                   weights=cnn_model.layers[2].get_weights(),
                                                   trainable=False)))
    sensor_model.add(TimeDistributed(MaxPooling1D(pool_length=8, stride=None, border_mode='valid')))
    # sensor_model.add(TimeDistributed(Dropout(dropout_rate)))

    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size * 2, 3,
                                                   border_mode='same',
                                                   activation='relu',
                                                   weights=cnn_model.layers[5].get_weights(),
                                                   trainable=False)))
    sensor_model.add(TimeDistributed(Convolution1D(hidden_layer_size * 2, 3,
                                                   border_mode='same',
                                                   activation='relu',
                                                   weights=cnn_model.layers[7].get_weights(),
                                                   trainable=False)))
    sensor_model.add(TimeDistributed(MaxPooling1D(pool_length=8, stride=None, border_mode='valid')))
    # sensor_model.add(TimeDistributed(Dropout(dropout_rate)))
    sensor_model.add(TimeDistributed(Flatten()))
    # sensor_model.add(TimeDistributed(Dense(hidden_layer_size*8, activation=activation)))
    # sensor_model.add(TimeDistributed(Dense(hidden_layer_size*8, activation=activation)))
    # sensor_model.add(TimeDistributed(Dense(3, activation=activation)))

    # state_input = Input(shape=(None, state_input_dim))
    # state_model = Model(state_input, state_input)

    # driving_model = Sequential()
    # driving_model.add(Merge([sensor_model, state_model], mode='concat'))
    sensor_model.add(LSTM(128, return_sequences=True))
    sensor_model.add(LSTM(128, return_sequences=True))
    sensor_model.add(TimeDistributed(Dense(3, activation='linear')))
    sensor_model.summary()
    
    return sensor_model


if __name__ == '__main__':
    
    # Load data
    # (x_dim, train_x, train_y, test_x, test_y) = load_data_cnn(FILE_PATH, QUADR_THETA)
    (x_dim, train_x, train_z, train_y, test_x, test_z, test_y) = load_data_rnn(FILE_PATH)

    # Load model from existing data
    # Load json and create model
    json_file = open('data/model_' + MODEL_NAME + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into new model
    model.load_weights('data/model_' + MODEL_NAME + '.h5')
    print('Loaded model from disk')
    
    # model.summary()
    
    model = update_model_to_rnn_sensor_only(model, x_dim, activation=ACT_FUNCTION,
                                            hidden_layer_size=NUM_HIDDEN_UNITS,
                                            dropout_rate=DROPOUT_RATE)
        
    # Optimizers
    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    rms = RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-08, decay=DECAY)

    # Compile model
    model.compile(loss='mean_absolute_error', metrics=['accuracy'],
                  optimizer=rms)
    
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
    
    # history = model.fit([train_x, train_z], train_y, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
    #                     validation_data=([test_x, test_z], test_y), shuffle=True)
    history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
                        validation_data=(test_x, test_y), shuffle=True)

    # Save eval-data
    sio.savemat('data/eval-data_rnn_' + TEST_NUM + '.mat', history.history)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('data/model_rnn_' + TEST_NUM + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('data/model_rnn_' + TEST_NUM + '.h5')
    print('Saved model to disk')

