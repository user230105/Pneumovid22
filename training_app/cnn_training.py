#IMPORTS*********************************************************************************
#standard lib
import argparse
import configparser
import os

#other imports
import numpy as np
import matplotlib.pyplot as plt

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

img_batch = int(config['train']['image_batch'])  #number of image per batch
#***********************************************************************************************************

#Preparing Data Seccion *****************************************************************
#constructor for a generator for image data augmentation

#train dataset
print('loading training dataset')
train_ds = keras.utils.image_dataset_from_directory(dataset_train_path,
                                                    image_size=(image_height, image_width),
                                                    color_mode='grayscale',
                                                    label_mode = "binary",
                                                    batch_size = img_batch,
                                                    seed=123
                                                    )
#Constructing validation dataset
print('loading validation dataset')
val_ds = keras.utils.image_dataset_from_directory(dataset_validation_path,
                                                    image_size=(image_height, image_width),
                                                    color_mode='grayscale',
                                                    label_mode = "binary",
                                                    batch_size = img_batch,
                                                    seed=123
                                                    )
#Constructing test dataset
print('loading test dataset')
test_ds = keras.utils.image_dataset_from_directory(dataset_test_path,
                                                    image_size=(image_height, image_width),
                                                    color_mode='grayscale',
                                                    label_mode = "binary",
                                                    batch_size = img_batch,
                                                    seed=123
                                                    )

print('classes train:', train_ds.class_names)

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
loss_function = keras.losses.BinaryCrossentropy(from_logits = False) 
metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()] 

optimizer = keras.optimizers.Adam(learning_rate=0.001)

#Regularizer
l2 = keras.regularizers.l2
l1 = keras.regularizers.l1


# Convolutional ***************************************************************************
cnn_model.add(layers.Rescaling(1./255, input_shape = (image_height, image_width, 1)))
cnn_model.add(layers.Conv2D(80, (6, 6), activation='relu', use_bias=True))
#cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Conv2D(112, (4, 4), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Dropout(0.2))
cnn_model.add(layers.Conv2D(144, (3, 3), activation='relu'))
cnn_model.add(layers.Conv2D(178, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Conv2D(204, (3, 3), activation='relu'))
cnn_model.add(layers.Conv2D(224, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Dropout(0.1))
cnn_model.add(layers.Conv2D(272, (3, 3), activation='relu'))
cnn_model.add(layers.Conv2D(328, (2, 2), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
#***************************************************************************************
#Deep neural network *******************************************************************
n_classes = len(train_ds.class_names)
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dropout(0.2))
cnn_model.add(layers.Dense(72, activation='relu'))
cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Dense(96, activation='relu'))
cnn_model.add(layers.Dense(170, activation='relu'))
cnn_model.add(layers.Dropout(0.2))
cnn_model.add(layers.Dense(1, activation='sigmoid'))
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
#early stoping when validation accuracy is max for two epoch
es = keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', 
                                    min_delta = 0.01, patience = 2, verbose=1)

flip_translate =  keras.Sequential([
                                        layers.RandomTranslation(height_factor = 0.08, 
                                        width_factor = 0.08,
                                        ),
                                        layers.RandomZoom(height_factor = 0.08, 
                                        width_factor = 0.08,
                                        ),
                                        layers.RandomFlip(mode="horizontal")
                                        ])

try:
    #data augmentation
    aug_ds = train_ds.map(lambda x, y: (flip_translate(x, training=True), y))
    #model training
    history = cnn_model.fit(aug_ds, epochs = epoch, verbose = 1, validation_data = val_ds, callbacks = [es])
    
    # summarize history for loss
    u_models.plot_training(history, 'loss')
    
    # summarize history for accuracy
    u_models.plot_training(history, 'binary_accuracy')   
    
    #evaluation***************************************************************************************
    print('Evaluating model')
    
    performance = cnn_model.evaluate(test_ds, verbose = 2)

    
    #saving evaluation result in format of config file    
    evaluation_file_path = os.path.join(models_path, model_name + '.result')
    label_map = {}
    for i in range(n_classes):
        label_map[train_ds.class_names[i]] = i
    u_models.save_eval_results(performance, label_map, evaluation_file_path)
    
    #saving model
    print('Saving model')
    cnn_model.save(os.path.join(models_path, model_name + '.h5'))

except (RuntimeError) as rt_error:
    print(rt_error)
except ValueError as v_error:
    print(v_error) 
