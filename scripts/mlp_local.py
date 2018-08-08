

from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
import numpy as np
import scipy.io as sio


##### Define Parameters
# Path to the input data-file
FILE_PATH = 'data/raw_sensor_data_01.mat'
# Percentage of data being training data
TRAIN_PERC = 0.9
# Training
LEARNING_RATE = 0.01
DECAY = 0.005
DROPOUT_RATE = 0.25
BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 512
NUM_EPOCHS = 2000
BETA = 0.0
QUADR_THETA = False
ACT_FUNCTION = 'relu'
TEST_NUM = '02'


##### Define Functions
# Load the data out of a .mat file
def load_data(file, quadr_theta):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['data_sensor'], float)
    data_y = np.asarray(data_set['data_pose'], float)
    
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
    
    return (train_x.shape[1], train_x, train_y, test_x, test_y)


# Create a model with just fully connected layers
def load_model_mlp(input_dim, l2_reg=0.01, activation='sigmoid',
                   hidden_layer_size=256, hidden_layer_num=3,
                   dropout_rate=0.25):
    print('Building MLP-Model.')
    
    model = Sequential()
    model.add(Dense(hidden_layer_size, W_regularizer=l2(l2_reg),
                    activity_regularizer=activity_l2(l2_reg),
                    init='lecun_uniform', input_dim=input_dim))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(hidden_layer_size, W_regularizer=l2(l2_reg),
                    activity_regularizer=activity_l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(hidden_layer_size, W_regularizer=l2(l2_reg),
                    activity_regularizer=activity_l2(l2_reg),
                    init='lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(hidden_layer_size/4, W_regularizer=l2(l2_reg),
                    activity_regularizer=activity_l2(l2_reg),
                    init = 'lecun_uniform'))
    model.add(BatchNormalization(epsilon=0.001, mode=0))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate*2))
    
    # model.add(Dense(hidden_layer_size/8, W_regularizer=l2(l2_reg), activity_regularizer=activity_l2(l2_reg)))
    # model.add(Activation(activation))
    # model.add(Dropout(dropout_rate))
    # model.add(Dense(hidden_layer_size/16))
    # model.add(Activation(activation))
    # model.add(Dense(hidden_layer_size/8))
    # model.add(Activation(activation))
    # model.add(Dense(hidden_layer_size/8))
    # model.add(Activation(activation))
    # model.add(Dense(hidden_layer_size/8))
    # model.add(Activation(activation))

    # model.add(Dropout(0.5))
    
    model.add(Dense(3))
    
    model.summary()
    
    return model


# Create a model with convolutional and maxPool layers
def load_model_cnn(input_dim, data_set_size, activation='relu', hidden_layer_size=512, hidden_layer_num=3):
    print('Building CNN-Model.')
    
    model = Sequential()
    model.add(Convolution1D(hidden_layer_size/2, 5, border_mode='same', input_shape=(input_dim, 1)))
    model.add(Activation(activation))
    model.add(Convolution1D(hidden_layer_size/2, 5, border_mode='same'))
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
    model.add(Dropout(0.25))
    
    model.add(Convolution1D(hidden_layer_size, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(Convolution1D(hidden_layer_size, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
    model.add(Dropout(0.25))
    
    # model.add(Convolution1D(128, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(Convolution1D(128, 3, border_mode='same'))
    # model.add(Activation(activation))
    # model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
    # model.add(Dropout(0.25))
    
    # activation = 'tanh'
    
    model.add(Flatten())
    # model.add(Dense(hidden_layer_size*2))
    # model.add(Activation(activation))
    model.add(Dense(hidden_layer_size))
    model.add(Activation(activation))
    model.add(Dense(hidden_layer_size/2))
    model.add(Activation(activation))
    model.add(Dense(hidden_layer_size/4))
    model.add(Activation(activation))
    model.add(Dropout(0.25))
    model.add(Dense(3))
    
    model.summary()
    
    return model


if __name__ == '__main__':

    # Load data
    (x_dim, train_x, train_y, test_x, test_y) = load_data(FILE_PATH, QUADR_THETA)
    
    # Load layers for model
    model = load_model_mlp(input_dim=x_dim, hidden_layer_size=NUM_HIDDEN_UNITS,
                           hidden_layer_num=10, l2_reg=BETA,
                           dropout_rate=DROPOUT_RATE, activation=ACT_FUNCTION)
    

    # Load json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    # model.load_weights("data/model_mlp_01.h5")
    # print("Loaded model from disk")
    
    # Optimizers
    # sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    
    # Compile model
    model.compile(loss='mean_absolute_error', #metrics=['accuracy'],
                  optimizer=adam)
    
    print('Start training MLP.')

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
    sio.savemat('data/eval-data_mlp_'+TEST_NUM+'.mat', history.history)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('data/model_mlp_'+TEST_NUM+'.json', 'w') as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights('data/model_mlp_'+TEST_NUM+'.h5')
    print('Saved model to disk')

