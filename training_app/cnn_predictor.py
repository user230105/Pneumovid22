from tensorflow import keras
import numpy as np
import cv2

import configparser
import os


def load_image(img_path, mode):

    try:
        if mode == 'gray':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)       
            return img
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)       
            return img
    except:
        print('Error loading image: ', img_path)
        return None

def enhance_image(img, shape, max_range = 255):
    ''' CXR Image Enhance Function
    based on papper

    Parameters:
    -----------
    img: numpy ndarray 
        Image in array of bytes format
    
    shape: tuple
        A tuple (width, height) containig the desire image's output shape

    max_range: int
        value between 1-255 for image normalization. Default 180.

    Returns:
    -------
     enhimage: numpy ndarray
    '''         
       
    #normalizing image
    enhimage = normalize_image(img, max_range) 
    # aplying clahe algoritms
    enhimage = clahe_filtering(enhimage)
    #resizing image
    enhimage = cv2.resize(enhimage, shape)
    return enhimage

def normalize_image(img, max_range = 255):
    ''' normalization function    

    Parameters:
    -----------
    img: numpy ndarray 
        Image in array of bytes format
    
    max_range: int
        value between 1-255 for image normalization.  Default 255.

    Returns:
    -------
     img: numpy ndarray
    '''    
    refimage = np.zeros(img.shape)          #reference image for normalization

    if max_range < 1:
        max_range = 1
    if max_range > 255:
        max_range = 255

    #normalizing image
    return cv2.normalize(img, refimage, 0, max_range, cv2.NORM_MINMAX)

def clahe_filtering(img):
    ''' function to apply clahe algoritms   

    Parameters:
    -----------
    img: numpy ndarray 
        Image in array of bytes format

    Returns
    -------
     img: numpy ndarray
    ''' 
    clahe = cv2.createCLAHE(clipLimit = 10) #clahe filter   
    # aplying clahe algoritms
    return clahe.apply(img)

config = configparser.ConfigParser()

#Getting configuration ******************************************************************
config.read('cnn_training.conf')

#models directory
basedir = os.getcwd()
models_path = os.path.join(basedir,config['paths']['models'])
model_name = 'model25'
cnn_model_path = os.path.join(models_path, model_name + '.h5')

#defining images size ***************************************************************************************
image_width = int(config['image']['width'])
image_height = int(config['image']['height'])
imshape = (image_height, image_width)

def load_model():
    #Loading Model
    print('loading model')
    if os.path.exists(cnn_model_path):
        try:             
            cnn_model = keras.models.load_model(cnn_model_path)                       
            return cnn_model
        except Exception as e:
            print(e)
            return None
    else:
        return None

model = load_model()

if model is None:
    exit('Model %s not found' %{model_name})
else:
    print('Model Loaded')

#loading test input tensor
img = load_image('test.png','gray')
x=[]
x.append(enhance_image(img, imshape))
X = np.array(x)

prediction = model.predict(X)

print(prediction)
