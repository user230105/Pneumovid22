print('starting')
from tensorflow import keras
from keras.utils.vis_utils import plot_model

import configparser
import os

config = configparser.ConfigParser()

#Getting configuration ******************************************************************
config.read('cnn_training.conf')
print('starting2')

#models directory
basedir = os.getcwd()
models_path = os.path.join(basedir,config['paths']['models'])
model_name = 'model25'
cnn_model_path = os.path.join(models_path, model_name + '.h5')

#Loading Model
print('loading model')

if os.path.exists(cnn_model_path):
    try:  
        print(cnn_model_path)           
        cnn_model = keras.models.load_model(cnn_model_path)     
        plot_model(cnn_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)  
        print('done')
    except Exception as e:
        print(e)

else:
    print('model missing')