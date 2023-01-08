#IMPORTS*********************************************************************************
#standard lib
import configparser
import os
from copy import deepcopy

#other imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils.dataset as dt
import utils.image as img

#****************************************************************************************

#parsers ********************************************************************************
config = configparser.ConfigParser()

print('Starting...')

#Getting configuration ******************************************************************
config.read('m_dataset_generator.conf')

#path to bimcv dataset
covidx_path = os.path.abspath(config['paths']['covidx_cxr2'])
#script directory
basedir = os.path.abspath(os.path.dirname(__file__))

#path to folder containing negative partition
covidx_datasets_test_part = os.path.join(covidx_path, 'test')
#path to folder containing positive partition
covidx_datasets_train_part = os.path.join(covidx_path, 'train')

#path to neg labels file
covidx_test_labels_path = os.path.join(covidx_path, config['labels']['test_label'])
#path to posi labels file
covidx_train_labels_path = os.path.join(covidx_path, config['labels']['train_label'])
#******************************************************************************************

try:
    #LABELS SECTION *************************************************************************
    dataset_test_labels = pd.read_csv(covidx_test_labels_path, sep=' ', names=['study', 'image', 'class', 'origin'])
    dataset_train_labels = pd.read_csv(covidx_train_labels_path, sep=' ', names=['study', 'image', 'class', 'origin'])    
    print('Labels loaded.')   
    #****************************************************************************************
    
    
    #Walking through all subject folder an making master dataframe********************     
    dataset_test_dict = dt.scan_covidx_dataset(covidx_datasets_test_part, 'test')

    dataset_train_dict = dt.scan_covidx_dataset(covidx_datasets_train_part, 'train')
        
    #*********************************************************************************
    #Labeling ************************************************************************ 
    print('Start labeling test partition dataset')    
    dataset_test_dict['class'] = [None] * len(dataset_test_dict['images'])

    for i in range(len(dataset_test_dict['images'])):  

        img_name = dataset_test_dict['images'][i]            
        image_row = dataset_test_labels.loc[dataset_test_labels['image'].str.contains(img_name, case=False)] 
        if len(image_row) == 1:
            label_cell = image_row['class']   
            dataset_test_dict['class'][i] = label_cell.values[0]    
        else:
            dataset_train_dict['class'][i] = 'none'  

    print('Start labeling train partition dataset')     
    dataset_train_dict['class'] = [None] * len(dataset_train_dict['images'])
    
    for i in range(len(dataset_train_dict['images'])):        
             
        img_name = dataset_train_dict['images'][i]    
        image_row = dataset_train_labels.loc[dataset_train_labels['image'].str.contains(img_name, case=False)]
        if len(image_row) == 1:
            label_cell = image_row['class']   
            dataset_train_dict['class'][i] = label_cell.values[0]  
        else:
            dataset_train_dict['class'][i] = 'none'
    #**********************************************************************************
        
    #**********************************************************************************
    print('Creating dataset images folder')  

    build_folder_path = os.path.join(basedir, config['datasets']['build'])
    dt.create_folder_struct(build_folder_path)
       
    #defining images size ***************************************************************************************
    image_width = int(config['image']['width'])
    image_height = int(config['image']['height'])
    imshape = (image_width, image_height)
    img_resolution = str(imshape[0]) + 'x' + str(imshape[1]) #string rerpresenting image resolution
    #***********************************************************************************************************
    #summary dict
    sumary_dict = {
        'image':[],
        'partition':[],
        'resolution':[],
        'label' : []                
    }

    #Getting images for train partition
    print('Starting train construction from covidx_cxr2 dataset')
    for i in range(len(dataset_train_dict['images'])):
        #calc image abs path
        image_name = dataset_train_dict['images'][i]
        class_name = dataset_train_dict['class'][i]

        image_abs_path = os.path.join(covidx_datasets_train_part, image_name)  
        img_loaded = img.load_image(image_abs_path, 'gray')
        
        if isinstance(img_loaded, np.ndarray) and class_name != 'none':           
            image_enhanced = img.enhance_image(img_loaded, imshape)
            image_save_path = os.path.join(build_folder_path,'train', class_name)

            if os.path.exists(image_save_path) == False:
                os.mkdir(image_save_path)

            success = img.save_image(os.path.join(image_save_path, image_name), image_enhanced) 

            if success:                
                sumary_dict['image'].append(image_name) #add image name to dict
                sumary_dict['partition'].append('train') #add image name to dict
                sumary_dict['resolution'].append(img_resolution)
                sumary_dict['label'].append(class_name)
    
    print('Starting test and validation construction from covidx_cxr2 dataset')
    for i in range(len(dataset_test_dict['images'])):
         #calc image abs path
        image_name = dataset_test_dict['images'][i]
        class_name = dataset_test_dict['class'][i]

        image_abs_path = os.path.join(covidx_datasets_test_part, image_name)  
        img_loaded = img.load_image(image_abs_path, 'gray')      
       
        if isinstance(img_loaded, np.ndarray):

            image_enhanced = img.enhance_image(img_loaded, imshape)
            if i < int(0.5 * len(dataset_test_dict['images'])):                
                image_save_path = os.path.join(build_folder_path,'test', class_name) 
                part_str = 'test'
            else:                
                image_save_path = os.path.join(build_folder_path,'validation', class_name)  
                part_str = 'validation'

            if os.path.exists(image_save_path) == False:
                os.mkdir(image_save_path)

            success = img.save_image(os.path.join(image_save_path, image_name), image_enhanced)

            if success:                
                sumary_dict['image'].append(image_name) #add image name to dict
                sumary_dict['partition'].append(part_str) #add image name to dict
                sumary_dict['resolution'].append(img_resolution)
                sumary_dict['label'].append(class_name)
            
    #saving tsv    
    sumary_dict_df = pd.DataFrame(sumary_dict) 
    sumary_dict_df.to_csv(os.path.join(build_folder_path, 'covidx_cxr2_dataset.tsv'), '\t')


except FileNotFoundError as e:
    print(e)
except KeyError as ke:
    print(ke)

