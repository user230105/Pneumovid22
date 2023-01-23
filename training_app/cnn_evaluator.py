#IMPORTS*********************************************************************************
#standard lib
import argparse
from concurrent.futures import process
import configparser
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#other imports
import numpy as np
import pandas as pd


from tensorflow import keras
from tensorflow.keras import layers, models, metrics

#parsers
console_parser = argparse.ArgumentParser()
config = configparser.ConfigParser()

#Console arguments definition------------------------------------------------------

#positional arguments
console_parser.add_argument('model', metavar="model_name", help="model's name")

console_parser.add_argument('dataset', metavar="dataset folder path", help="path to folder with dataset")


args = console_parser.parse_args()

#Getting configuration ******************************************************************
config.read('cnn_training.conf')

#****************************************************************************************
basedir = os.path.abspath(os.path.dirname(__file__))
dataset_train_path = os.path.join(basedir, args.dataset,'train')
dataset_test_path = os.path.join(basedir, args.dataset,'test')
dataset_val_path = os.path.join(basedir, args.dataset,'validation')

#models directory
models_path = os.path.join(basedir,config['paths']['models'])
model_name = args.model

#model path
cnn_model_path = os.path.join(models_path, model_name + '.h5')

#defining images size ***************************************************************************************
image_width = int(config['image']['width'])
image_height = int(config['image']['height'])
imshape = (image_height, image_width)
img_batch = int(config['train']['image_batch'])

#Loading Model
if os.path.exists(cnn_model_path):
    try:   
        print('Loading Model') 
        cnn_model = keras.models.load_model(cnn_model_path)
        #Load Dataset ***************************************************************************************
        eval_train_ds = keras.utils.image_dataset_from_directory(dataset_train_path,
                                                         image_size=imshape, 
                                                         color_mode='grayscale',
                                                         label_mode = "int",
                                                         batch_size=img_batch)

        eval_test_ds = keras.utils.image_dataset_from_directory(dataset_test_path,
                                                         image_size=imshape, 
                                                         color_mode='grayscale',
                                                         label_mode = "int",
                                                         batch_size=img_batch)                                                 

        eval_val_ds = keras.utils.image_dataset_from_directory(dataset_val_path,
                                                         image_size=imshape, 
                                                         color_mode='grayscale',
                                                         label_mode = "int",
                                                         batch_size=img_batch)
        print(eval_train_ds)

        #evaluating agains train partition  
        print('evaluating again train partition')                                               
        performance_train = cnn_model.evaluate(eval_train_ds, verbose = 2)  

        #evaluating agains test partition  
        print('evaluating again test partition')                                               
        performance_test = cnn_model.evaluate(eval_test_ds, verbose = 2)

        #evaluating agains validation partition  
        print('evaluating again validation partition')                                               
        performance_val = cnn_model.evaluate(eval_val_ds, verbose = 2)    

        """   
        y_predict= cnn_model.predict(eval_ds)

        eval_ds_unbatch = eval_ds.unbatch()  
        images, labels = tuple(zip(*eval_ds_unbatch))
        y_true = np.array(labels)
        
        #metrics
        m_bin_acc = metrics.BinaryAccuracy()
        m_precision = metrics.Precision()
        m_recall = metrics.Recall()

        #metrics calculation
        m_bin_acc.update_state(y_true, y_predict)
        m_precision.update_state(y_true, y_predict)
        m_recall.update_state(y_true, y_predict)


        binary_accuracy = m_bin_acc.result().numpy()
        precision = m_precision.result().numpy()
        recall = m_recall.result().numpy()

        print('metrics ', binary_accuracy, precision, recall) """

    except Exception as e:
        exit(print(e))
else:     
    exit('Model %s not found' %{args.model})   

   