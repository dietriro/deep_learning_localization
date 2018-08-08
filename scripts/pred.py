
from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
import numpy as np
import scipy.io as sio


FILE_PATH = 'data/raw_sensor_data_01.mat'
TRAIN_PERC = 0.9


# Load the data out of a .mat file
def load_data(file, quadr_theta=False, train_perc=0.9):
    # Read data from .mat file
    data_set = sio.loadmat(file)
    data_x = np.asarray(data_set['data_sensor'], float)
    data_y = np.asarray(data_set['data_pose'], float)
    
    # Randomly permute data
    perm = np.random.permutation(data_x.shape[0])
    data_x = data_x[perm]
    data_y = data_y[perm]
    
    # Add bias
    bias = np.ones((data_x.shape[0], 1))
    data_x = np.append(bias, data_x, 1)
    
    # Split data into train and test data
    train_size = int(len(perm) * train_perc)
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


if __name__ == '__main__':
    
    model_name = 'mlp_01'
    
    # Load data
    (x_dim, train_x, train_y, test_x, test_y) = load_data(FILE_PATH)
    
    # Load json and create model
    json_file = open('data/model_'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into new model
    model.load_weights('data/model_'+model_name+'.h5')
    print('Loaded model from disk')
    
    print(train_x[2, :].transpose().shape)
    
    pred = model.predict(test_x[2, :].reshape((1, 668)), batch_size=1, verbose=1)
    
    print(pred)
    print(test_y[2, :])