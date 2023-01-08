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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils.models as u_models

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
       #constructor for a generator for image data augmentation
        train_datagen = ImageDataGenerator(rescale=1./255,
                                            width_shift_range=[-20,20],
                                            height_shift_range=[-20,20], 
                                            zoom_range=0.1,
                                            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        #train generator
        img_batch = int(config['train']['image_batch'])  #number of image per batch
        train_generator = train_datagen.flow_from_directory(dataset_train_path,
                                                            target_size=(image_height, image_width),
                                                            color_mode='grayscale',
                                                            class_mode = "categorical",
                                                            batch_size = img_batch
                                                            )
        #Constructing validation dataset
        val_generator = test_datagen.flow_from_directory(dataset_validation_path,
                                                            target_size=(image_height, image_width),
                                                            color_mode='grayscale',
                                                            class_mode = "categorical",
                                                            batch_size = img_batch
                                                            )
        #Constructing test dataset
        test_generator = test_datagen.flow_from_directory(dataset_test_path,
                                                            target_size=(image_height, image_width),
                                                            color_mode='grayscale',
                                                            class_mode = "categorical",
                                                            batch_size = img_batch
                                                            )
        
        #Edit model Segment************************************************************************
        loss_function = keras.losses.CategoricalCrossentropy(from_logits = False) 
        metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()] 

        optimizer = keras.optimizers.Adam(learning_rate=0.00001)

        cnn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

        #training
        print('Starting Training ...')

        #Fit parameters
        epoch = int(config['train']['epoch'])  #number of epoch
        steps_epoch = int(train_generator.samples/img_batch)
        val_steps = int(val_generator.samples/img_batch)

        history = cnn_model.fit(train_generator, epochs = epoch,
                            steps_per_epoch = steps_epoch, verbose = 1, shuffle=True,
                            validation_data = val_generator, validation_steps = val_steps)

        # summarize history for loss
        u_models.plot_training(history, 'loss')

        # summarize history for accuracy
        u_models.plot_training(history, 'binary_accuracy') 

        #evaluation***************************************************************************************
        print('Evaluating model')
        eval_steps = int(test_generator.samples/img_batch)
        performance = cnn_model.evaluate(test_generator, steps = eval_steps, verbose = 2)

        #saving model
        print('Saving model')
        cnn_model.save(os.path.join(models_path, 'fine_' + model_name + '.h5'))

    except Exception as e:
        exit(print(e))
else:     
    exit('Model %s not found' %{args.model})