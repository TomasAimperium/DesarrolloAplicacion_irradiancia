from numpy import sum,array,median,abs,nan,ndarray,concatenate,mean,max,var,std,isnan

#import pandas as pd
from scipy.signal import savgol_filter
import datetime
from scipy.stats import median_absolute_deviation
import ast
import logging
import json
from config import *
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')  
from numpy import sum,array,median,abs,nan,ndarray,concatenate,mean,max,var,std,isnan,cumsum
import datetime
import logging
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.models import load_model
from datetime import datetime,timedelta




def mean_squared_error(ypred,yreal):
    if len(ypred) == len(yreal):
        MSE = sum((ypred - yreal)**2)/len(ypred)
    else:
        MSE = 0
    return MSE


def load_models():
    '''
    Funcion que carga los modelos de maximo, media y regresion.
    '''
    disable_eager_execution()
    models = ["bhi_model","dhi_model"]
    M = []
    for i in models:
        try:
            M.append(load_model(f'model/{i}.h5'))
        except:
            logging.error('AI_error_5: modelo'+ str(i) +'no encontrado') 
    return M

M =  load_models()
    
def predictions(inp):

    val_time = inp['time'][:N]
    val_data_fuzzy = inp['fuzzy'][:N].reshape(-1,1).reshape(1,N,1)
    val_data_direct = inp['direct'][:N].reshape(-1,1).reshape(1,N,1)
    pre_time = inp['time'][N:]
    pre_data_fuzzy = inp['fuzzy'][N:].reshape(-1,1).reshape(1,N,1)
    pre_data_direct =inp['direct'][N:].reshape(-1,1).reshape(1,N,1)
    
    
    #predicciones 
    directa,difusa = M[0].predict(pre_data_direct),M[1].predict(pre_data_fuzzy)
    directa[directa<0.05] = 0
    difusa[difusa<0.05] = 0
    cs = cumsum(directa + difusa)
    
    
    #validacion
    directa_,difusa_ = M[0].predict(val_data_direct),M[1].predict(val_data_fuzzy)
    directa_[directa_<0.05] = 0
    difusa_[difusa_<0.05] = 0
    cs_ = cumsum(directa_ + difusa_)
    total_ = directa_ + difusa_   
    error_reg = mean_squared_error(total_,val_data_fuzzy[:n]+val_data_direct[:n])
    error_sum = mean_squared_error(cs_,cumsum(val_data_fuzzy.reshape(-1)[:n]+val_data_direct.reshape(-1)[:n]))
    
    #time_reg
    now = datetime.strptime(pre_time[-1], '%Y-%m-%d %H:%M:%S')
    date_list = [str(now + timedelta(minutes=x)) for x in range(1,n)]
    
    output = {
        'header': inp['header'],
        'update_time':str(pre_time[-1]),
        'inputs':{
            'prepro':{
                'Directa': str(list(inp['direct'])),
                'Difusa': str(list(inp['fuzzy']))
            },
            'date_input': str(list(inp['time']))
        },
        
        'output':{
          'Regression':{
              "Directa":str(list(directa[0])),
              "Difusa":str(list(difusa[0]))
          },
          'CumSum':str(list(cs)),
          'time_reg': str(date_list),
          'validation':{
            'check': "yes",
            'MAPEreg':str(error_reg),
            "MAPEsum":str(error_sum)
            }
        },
        
        'params': {
            'smooth':str(SAVGOL),
            'samp': str(SAMPLEO)},
        "QoI": {
            "Completeness": "1",
            "Outliers": "0"}
    }
        
        
    return output