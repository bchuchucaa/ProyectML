import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image
from keras import backend as k
from apiSNN.models import Image

def predecir_imagen(file):
  url_modelo= r'apiSNN/Logica/modelo'
  url_pesos = r'apiSNN/Logica/pesos'
  print(' -- INTENTA CARGAR -- ' * 5)

  loaded_model = cargar_rnn(url_modelo, url_pesos)
  x = load_img(file, target_size=(100, 100))
  x=img_to_array(x)
  x=np.expand_dims(x, axis=0)
  arreglo= loaded_model.predict(x)
  resultado= arreglo[0]
  maxElement = np.amax(resultado)
  MaxElement='%.2f'%(maxElement*100)
  respuesta=np.argmax(resultado)
  print("Con una probabilidad de "+MaxElement+"% esta: ")
  if respuesta==0:
    respuesta='Enojado'
  elif respuesta==1:
    respuesta='Asco'
  elif respuesta==2:
    respuesta='Miedo'
  elif respuesta==3:
    respuesta='Feliz'
  elif respuesta==4:
    respuesta='Triste'
  elif respuesta==5:
    respuesta='Sorprendido'
  elif respuesta==6:
    respuesta='Neutral'
  predecido=Image()
  datos = dict()
  datos['pred'] = respuesta
  datos['prob'] = MaxElement
  imagenes=Image(2,file,respuesta,MaxElement)
  imagenes.save()
  print(respuesta)
  print(MaxElement)
  return datos
   


def cargar_rnn(nombreArchivoModelo, nombreArchivoPesos):
    print('INICIA PROCES CARGA <---------------------------------------------------')
    k.reset_uids()
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo + '.json', 'r') as f:
        print('INTENTA LEER <<___ '*5)
        model = model_from_json(f.read())
        print('FINALIZA EL LEER <----'*10)
    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos + '.h5')
    print("Red Neuronal Cargada desde Archivo")
    return model
