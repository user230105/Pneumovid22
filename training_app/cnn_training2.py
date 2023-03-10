#IMPORTS*********************************************************************************
#standard lib
import argparse
import configparser
import os

#other imports
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.utils import np_utils
#from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils.models as u_models
#****************************************************************************************

#parsers
console_parser = argparse.ArgumentParser()
config = configparser.ConfigParser()

#Console arguments definition------------------------------------------------------

#positional arguments
console_parser.add_argument('model', metavar="model_name", help="model's name")

args = console_parser.parse_args()
#------------------------------------------------------------------------------------

#Getting configuration ******************************************************************
config.read('cnn_training.conf')

#script directory
basedir = os.path.abspath(os.path.dirname(__file__))

#path datasets
dataset_train_path = os.path.join(basedir, config['paths']['dataset'], 'train')
dataset_test_path = os.path.join(basedir, config['paths']['dataset'], 'test')
dataset_validation_path = os.path.join(basedir, config['paths']['dataset'], 'validation')


#models directory
models_path = os.path.join(basedir,config['paths']['models'])
model_name = args.model

#defining images size ***************************************************************************************
image_width = int(config['image']['width'])
image_height = int(config['image']['height'])
imshape = (image_width, image_height)
#***********************************************************************************************************

#Preparing Data Seccion *****************************************************************
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

print('classes train:', train_generator.class_indices)

""" for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
	# generate batch of images
    batch = train_generator.next()
	# convert to unsigned integers for viewing    
    batch_img = batch[0].astype('uint8')
    image = np.squeeze(batch_img,0)    
	# plot raw pixel data
    plt.imshow(image)
# show the figure
plt.show() """

#CNN Model**************************************************************************************************************
cnn_model = models.Sequential()

#Edit model Segment************************************************************************
loss_function = keras.losses.CategoricalCrossentropy(from_logits = False) 
metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()] 

optimizer = keras.optimizers.Adam(learning_rate=0.0001)

#Regularizer
l2 = keras.regularizers.l2
l1 = keras.regularizers.l1


# Convolutional ***************************************************************************
cnn_model.add(layers.Conv2D(72, (5, 5), activation='relu', 
                 input_shape = (image_height, image_width, 1), use_bias=True))
cnn_model.add(layers.MaxPooling2D((2, 2)))
#cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Conv2D(104, (4, 4), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
#cnn_model.add(layers.Dropout(0.2))
cnn_model.add(layers.Conv2D(136, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(170, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
#cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Conv2D(204, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
#cnn_model.add(layers.Dropout(0.1))
cnn_model.add(layers.Conv2D(264, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(320, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
#***************************************************************************************
#Deep neural network *******************************************************************
n_classes = len(train_generator.class_indices)
cnn_model.add(layers.Flatten())
#cnn_model.add(layers.Dropout(0.2))
cnn_model.add(layers.Dense(80, activation='relu'))
cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Dense(170, activation='relu'))
cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Dense(350, activation='relu'))
#cnn_model.add(layers.Dropout(0.2))
cnn_model.add(layers.Dense(72, activation='relu'))
cnn_model.add(layers.Dense(n_classes, activation='softmax'))
#****************************************************************************************

cnn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

#Getting sumary file *******************************************************************
summary_file_path = os.path.join(models_path, model_name + '.sum')
with open(summary_file_path, 'w') as summary:
    cnn_model.summary(print_fn = lambda x: summary.write(x + '\n'))

#training
print('Starting Training ...')

#Fit parameters
epoch = int(config['train']['epoch'])  #number of epoch
steps_epoch = int(train_generator.samples/img_batch)
val_steps = int(val_generator.samples/img_batch)

#early stoping when validation accuracy is max for two epoch
es = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                    min_delta = 0, patience = 4, verbose= 1, mode="min", baseline= None, restore_best_weights=True)
try:
    history = cnn_model.fit(train_generator, epochs = epoch,
                            steps_per_epoch = steps_epoch, shuffle=True,
                            validation_data = val_generator, validation_steps = val_steps, callbacks = [es], verbose = 1)
    

    # summarize history for loss
    u_models.plot_training(history, 'loss')

    # summarize history for accuracy
    u_models.plot_training(history, 'binary_accuracy')   

    #evaluation***************************************************************************************
    print('Evaluating model')
    eval_steps = int(test_generator.samples/img_batch)
    performance = cnn_model.evaluate(test_generator, steps = eval_steps, verbose = 2)


    #saving evaluation result in format of config file    
    evaluation_file_path = os.path.join(models_path, model_name + '.result')
    u_models.save_eval_results(performance, train_generator.class_indices, evaluation_file_path)

    #saving model
    print('Saving model')
    cnn_model.save(os.path.join(models_path, model_name + '.h5'))

except (RuntimeError) as rt_error:
    print(rt_error)
except ValueError as v_error:
    print(v_error) 