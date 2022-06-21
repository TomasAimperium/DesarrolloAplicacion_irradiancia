import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import config as c

ep = c.EPOCHS
bs = c.BATCH_SIZE
n = c.n

callbacks = [
             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
            ]

def model_direct(train_X,train_Y):
    keras.backend.clear_session()
    model_ = Sequential([
        Conv1D(filters=128, kernel_size=10,strides=1, padding="causal",activation="relu",input_shape=[None, 1]), 
        LSTM(192, activation="tanh", return_sequences=False), 
        Dense(640, activation="relu"),
        Dense(n, activation="linear")]
        )
    model_.compile(loss = 'mse',
                   optimizer = 'adam')
    return model_  

def model_fuzzy(train_X,train_Y):
    keras.backend.clear_session()
    model_ = Sequential([
        Conv1D(filters=128, kernel_size=10,strides=1, padding="causal",activation="relu",input_shape=[None, 1]), 
        LSTM(192, activation="tanh", return_sequences=False), 
        Dense(640, activation="relu"),
        Dense(n, activation="linear")]
        )
    model_.compile(loss = 'mse',
                   optimizer = 'adam')
    return model_   


def fit_direct(model_direct,train_X, train_Y,test_X,test_Y):
    model_direct.fit(train_X, train_Y, epochs = ep, batch_size = bs, validation_data=(test_X, test_Y), verbose= 1,callbacks = callbacks)
    return model_direct


def fit_fuzzy(model_fuzzy,train_X, train_Y,test_X,test_Y):
    model_fuzzy.fit(train_X, train_Y, epochs = ep, batch_size = bs, validation_data=(test_X, test_Y), verbose= 1,callbacks = callbacks)
    return model_fuzzy