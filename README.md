# Aplicación de irradiancia.

Desarrollo del modulo de irradiancia, prediccion los valores de irradiancia para obtener la producción. Este sistema admite una serie de datos históricos de irradiancia y con el objetivo de entrenar una red neuronal CNN 1-D + LSTM, de forma que para obtener los valores futuros hará falta nada más que unos pocos valores en el pasado.   


Descripción de los archivos:

* config.py: en este archivo se dan los parámetros necesarios para la ejecución de la aplicación: franja horaria, tiempo de sampleo, datos empleados para el entrenamiento, índice de suavizado, datos necesarios para predecir, datos a predecir y parámetros de entrenamiento.

* model.py: este archivo guarda funciones que definene los modelos de Deep Learning que se entrenarán con los datos históricos. Hay 4 funciones en su interior: model_direct (definición del modelo para la radiación directa), model_fuzzy (definición modelo para la radiación difusa), fit_direct (función que entrena el modelo de radiación directa) y fit_fuzzy (función que entrena el modelo de radiación difusa).

* prepro.py: archivo que contiene funciones para el preprocesamiento de datos previo a la realización de la predicción. Cuenta con 3 funciones: interp (formateo del json de entrada), savgol (suavizado y elimninación del ruido de los datos) y pipeline (aplicación de las funciones anteriores).

* train.py: archivo que realiza el entremiento de los modelos. En primer lugar realiza el split de datos (train_split) y posteriormente se realizar el fit del modelo para después guardar el modelo.

* test.py: tests unitarios de la aplicación y comprobación de la calidad de actuación del modelo. Este archivo muestra las siguientes funciones:
  * load_historic: carga datos históricos de un json.
  * load_instance: carga un json de datos.
  * isnumeric: comprueba si un dato tiene un valor numérico.
  * isstring: comprueba si un dato es un string.
  * isdate: comprueba si un dato es una fecha.
  * error_pred: calcula el error de las últimas predicciones y lo compara con un valor.
  * test_train: compreba si los valores para el entrenamiento son adecuados.
  * test_format: comprueba si el formato de los datos es adecuado.
  * test_nan: comprueba si hay una cantidad excesiva de NaN en los datos.
  * test_prep: comprueba si el resultado del preprocesamiento es adecuado.
  * test_predict: comprueba si el resultado de las predicciones es adecuado.
  * test_error: compara errores con un valor umbral.

* prediction.py: este script sirve para hacer las predicciones de los valores de irradiancia. Este cuenta con 3 funciones:
  
  * mean_squared_error: calcula el error de las predicciones.
  * load_models: carga los modelos de tensorflow.
  * predictions: se formatean los datos y estos se incluyen en el modelo de predicción para que estos sean exportados a un json posteriormente.

* main.py: ejemplo de ejecucion de la aplicación.



Ejemplo de input:

{
    "header": {
        "empresa": "empresa",
        "id": "asdasdasdasd",
        "planta": "1234"
    },
    "update_time": "2022-04-25 11:46:33.759360",
    "data": {
       "datetime":  "['2022-04-25 11:43:52.967010', '2022-04-25 11:44:52.967026', ...
       "values":{
            "Directa": "[0.0012350676792239, 0.0003155238490482, -0.0008832816714252, -0.0021860300476059, ...
            "Difusa":  "[0.357546974330285, 0.3902486476135734, 0.4230178643234278, 0.4555569377266714, ... 
       }
    }
  }
}


Ejemplo de output:


 {
   "header": {
   "empresa": "empresa",
   "id": "asdasdasdasd",
   "planta": "1234"
   },
   'update_time':str(pre_time[-1]),
   'inputs':{
      'prepro':{
            "Directa": "[0.0012350676792239, 0.0003155238490482, -0.0008832816714252, -0.0021860300476059, ...
            "Difusa":  "[0.357546974330285, 0.3902486476135734, 0.4230178643234278, 0.4555569377266714, ... 
      },
      'date_input': "['2022-04-25 11:43:52.967010', '2022-04-25 11:44:52.967026', ...
   },

   'output':{
    'Regression':{
          "Directa": "[0.0012350676792239, 0.0003155238490482, -0.0008832816714252, -0.0021860300476059, ...
          "Difusa":  "[0.357546974330285, 0.3902486476135734, 0.4230178643234278, 0.4555569377266714, ... 
    },
    'CumSum':[0.0012350676792239, 0.12328754498, 0.22343467567, ...
    'time_reg': "['2022-04-25 11:43:52.967010', '2022-04-25 11:44:52.967026', ...
    'validation':{
      'check': "yes",
      'MAPEreg':0.89,
      "MAPEsum":0.75
      }
   },

   'params': {
      'smooth':31,
      'samp': 10},
   "QoI": {
      "Completeness": "1",
      "Outliers": "0"}
   }




