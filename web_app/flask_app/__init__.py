import configparser
import os


from flask import Flask
from tensorflow import keras
import numpy as np
from flask_app import image

app = Flask(__name__, static_url_path='')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

config = configparser.ConfigParser()
model_data = configparser.ConfigParser()

#Getting configuration ******************************************************************
config.read('app.conf')

#models directory
basedir = os.getcwd()
models_path = os.path.join(basedir,config['paths']['models'])
model_name = config['paths']['name']
cnn_model_path = os.path.join(models_path, model_name + '.h5')
cnn_model_result_path = os.path.join(models_path, model_name + '.result')

#defining images size ***************************************************************************************
image_width = int(config['image']['width'])
image_height = int(config['image']['height'])
imshape = (image_height, image_width)

#Set model excecution related functions to flask app********************************************************
ALLOWED_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.allowed_file = allowed_file
#************************************************************************************************************
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

app.model = load_model()
if app.model is None:
    exit('Model %s not found' %{model_name})
else:
    print('Model Loaded')
#*********************************************************************************************************

model_data.read(cnn_model_result_path)
labels = [None]*len(model_data['Classes'])

for label in model_data['Classes']:        
    label_value = int(model_data['Classes'][label])
    labels[label_value] = label

def get_label_sparse(prediction):
    y = prediction[0]        
    index_label = round(y[0])
    return labels[index_label] 

app.get_label_sparse = get_label_sparse

def get_label_categorical(prediction):
    y = prediction[0]   
    index_label = y.argmax()
    return labels[index_label]
    
app.get_label_categorical = get_label_categorical
#***************************************************************************************************************
def image_to_tensor(image_file):
    x=[]
    img = image.load_image(image_file)    
    x.append(image.enhance_image(img, imshape))
    return np.array(x)

app.get_input_tensor = image_to_tensor

from flask_app import routes