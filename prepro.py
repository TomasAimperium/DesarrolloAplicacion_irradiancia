import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
from numpy import sum,array,median,abs,nan,ndarray,concatenate,mean,max,var,std,isnan
from scipy.signal import savgol_filter
import ast
import datetime
import json
from config import *

def interp(js):
    return {'time':ast.literal_eval(js['data']['datetime']),
            'directa':json.loads(js['data']['values']['Directa']),
            'difusa':json.loads(js['data']['values']['Difusa'])}

def savgol(X,n):
    savgol_data = savgol_filter(X, n, 9)
    return savgol_data


def pipeline(data = None,SAVGOL= SAVGOL):
    header = data['header']
    data = interp(data)
    direct,fuzzy,time =  data['directa'],data['difusa'],data['time']
    if SAVGOL%2 == 0: SAVGOL += 1
    fuzzy_f = savgol(fuzzy,SAVGOL)
    direct_f = savgol(direct,SAVGOL)
    output = {
        'header': header,
        'time':time,
        'fuzzy':fuzzy_f,
        'direct':direct_f}
    return output