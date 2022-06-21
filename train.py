import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pandas as pd
import numpy as np
import config as c
from model import *
from prepro import *
from prediction import *


def train_split(data,n_past,n_future):
    y_col = 0
    _data = data.reshape(-1,1)
    data_X   = []
    data_Y   = []
    for i in np.arange(n_past, len(_data) - n_future + 1):
        data_X.append(_data[i-n_past:i,0:_data.shape[1]])
        data_Y.append(_data[i:i + n_future, 0])
    data_X, data_Y = np.array(data_X), np.array(data_Y)
    return data_X,data_Y

n_past,n_future = c.N,c.n
n_data = c.DATOS_TRAIN


#######directa#######
print("directa")
D = pd.read_csv('data/bhi_data.csv').values[:n_data,1]
data = savgol(D,c.SAVGOL)
data_X,data_Y = train_split(data,n_past,n_future)
train_X,train_Y,test_X,test_Y = data_X[:int(n_data*0.8)],data_Y[:int(n_data*0.8)],data_X[int(n_data*0.8):],data_Y[int(n_data*0.8):]
Md = model_direct(train_X,train_Y)
fit_direct(Md,train_X, train_Y,test_X,test_Y)
Md.save("model/Md_mal.h5")


#######difusa#######
print("difusa")
D = pd.read_csv('data/dhi_data.csv').values[:n_data,1]
data = savgol(D,c.SAVGOL)
data_X,data_Y = train_split(data,n_past,n_future)
train_X,train_Y,test_X,test_Y = data_X[:int(n_data*0.8)],data_Y[:int(n_data*0.8)],data_X[int(n_data*0.8):],data_Y[int(n_data*0.8):]
Mf = model_fuzzy(train_X,train_Y)
fit_fuzzy(Mf,train_X, train_Y,test_X,test_Y)
Mf.save("model/Mf_mal.h5")

#########################################################################################

