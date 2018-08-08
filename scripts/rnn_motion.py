from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
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
LEARNING_RATE = 0.001
DECAY = 0.0005  # after 2000 => lr=0.002
DROPOUT_RATE = 0.1
BATCH_SIZE = 10000  # 400 > 200 > 100 -> in the end
NUM_HIDDEN_UNITS = 64
NUM_EPOCHS = 2500
BETA = 0.000
QUADR_THETA = False
ACT_FUNCTION = 'sigmoid'
TEST_NUM = '06'  # Defines the name of the saved file
LOAD_EXISTING_MODEL = True  # Whether existing model should be loaded or new one compiled
MODEL_NAME = 'cnn_05'  # Name of existing model


##### Define Functions
# Load the data out of a .mat file
def load_data(file, quadr_theta):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['data_sensor'], float)
    data_y = np.asarray(data_set['data_pose'], float)
    
    data_x[data_x == 100.0] = 7.0
    
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


# Create a model with convolutional and maxPool layers
def load_model_cnn(input_dim, data_set_size, l2_reg=0.01, activation='relu',
                   hidden_layer_size=64, hidden_layer_num=3,
                   dropout_rate=0.25):
    print('Building CNN-Model.')
    
    model = Sequential()
    
    # Best so far: 1cnn with h/8, 3 and pl=16
    
    # Convolutional Layer 1
    model.add(Convolution1D(hidden_layer_size / 8, 3, border_mode='same',
                            input_shape=(input_dim, 1)))
    model.add(LSTM(3, init='glorot_uniform', inner_init='orthogonal',
                   forget_bias_init='one', activation='tanh',
                   inner_activation='hard_sigmoid', W_regularizer=None,
                   U_regularizer=None, b_regularizer=None,
                   dropout_W=0.2, dropout_U=0.2))
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
        print('Loaded model from disk')
    
    else:
        # Load layers for model
        model = load_model_cnn(x_dim, train_x.shape[0], activation=ACT_FUNCTION,
                               hidden_layer_size=NUM_HIDDEN_UNITS, l2_reg=BETA,
                               dropout_rate=DROPOUT_RATE)
    
    # Optimizers
    # sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    # adag = Adagrad(lr=LEARNING_RATE, epsilon=1e-08, decay=0.0)      # Not bad with lr=0.01
    adad = Adadelta(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
    
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
    sio.savemat('data/eval-data_cnn_' + TEST_NUM + '.mat', history.history)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('data/model_cnn_' + TEST_NUM + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('data/model_cnn_' + TEST_NUM + '.h5')
    print('Saved model to disk')

