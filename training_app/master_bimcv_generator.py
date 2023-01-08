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
bimcv_path = os.path.abspath(config['paths']['bimcv_covid19'])
#script directory
basedir = os.path.abspath(os.path.dirname(__file__))

#path to folder containing negative partition
bimcv_datasets_neg_part = os.path.join(bimcv_path, config['datasets']['part_neg'])
#path to folder containing positive partition
bimcv_datasets_posi_part = os.path.join(bimcv_path, config['datasets']['part_posi'])

#path to neg labels file
bimcv_neg_labels_path = os.path.abspath(os.path.join(bimcv_path, config['labels']['neg_label_path']))
#path to posi labels file
bimcv_posi_labels_path = os.path.abspath(os.path.join(bimcv_path, config['labels']['posi_label_path']))
#path to test result file
bimcv_posi_test_path = os.path.abspath(os.path.join(bimcv_path, config['labels']['test_result_path']))
#******************************************************************************************

try:
    #LABELS SECTION *************************************************************************
    dataset_neg_img_labels = pd.read_table(bimcv_neg_labels_path)
    dataset_posi_img_labels = pd.read_table(bimcv_posi_labels_path)
    dataset_posi_test = pd.read_table(bimcv_posi_test_path)
    print('Labels loaded.')
    #****************************************************************************************
    
    
    #Walking through all subject folder an making master dataframe********************     
    dataset_neg_dict = dt.scan_dataset(bimcv_datasets_neg_part, 'negative')

    dataset_posi_dict = dt.scan_dataset(bimcv_datasets_posi_part, 'positive')
        
    #*********************************************************************************
    #Labeling ************************************************************************ 
    print('Start labeling negative partition dataset')    
    dataset_neg_dict['class'] = [None] * len(dataset_neg_dict['image'])

    for i in range(len(dataset_neg_dict['image'])):        
        subject = dataset_neg_dict['PatientID'][i]
        session = dataset_neg_dict['ReportID'][i]       
        label = dt.get_labels_binary(subject, session, dataset_neg_img_labels, 'negative')        
        dataset_neg_dict['class'][i] = label   
    
    dataset_neg_dict = dt.discard_dict_entry(dataset_neg_dict)

    print('Start labeling positive partition dataset')     
    dataset_posi_dict['class'] = [None] * len(dataset_posi_dict['image'])
    
    for i in range(len(dataset_posi_dict['image'])):        
        subject = dataset_posi_dict['PatientID'][i]
        session = dataset_posi_dict['ReportID'][i]        
        label = dt.get_labels_binary(subject, session, dataset_posi_img_labels, 'positive')        
        dataset_posi_dict['class'][i] = label    
    
    dataset_posi_dict = dt.discard_dict_entry(dataset_posi_dict)
    
    #**********************************************************************************
    #Merging positive and negative dataset into one************************************ 
    master_cxr_dict = dt.merge_dict(dataset_neg_dict, dataset_posi_dict)    
    #**********************************************************************************
    #Making master data frame**********************************************    
    master_cxr_df = pd.DataFrame(master_cxr_dict)    
   

    print('Creating dataset images folder')  

    build_folder_path = os.path.join(basedir, config['datasets']['build'])
    dt.create_folder_struct(build_folder_path)

    images_names = master_cxr_dict['image']             #images path array
    partition = master_cxr_dict['partition']            #partition array
    subject = master_cxr_dict['PatientID']              #subjects array
    session  = master_cxr_dict['ReportID']              #session array
    class_labels = master_cxr_dict['class']             #covid positive or not classes
    index = [index for index in range(len(images_names))]       #index list
    
    #defining images size ***************************************************************************************
    image_width = int(config['image']['width'])
    image_height = int(config['image']['height'])
    imshape = (image_width, image_height)
    img_resolution = str(imshape[0]) + 'x' + str(imshape[1]) #string rerpresenting image resolution
    #***********************************************************************************************************
   
    #spliting dataset in train and test subset *******************************************
    train_size = float(config['datasets']['train_size'])
    #split the index list randomly in two diferent dataset
    index_train, index_vt, label_train, label_vt = train_test_split(index, class_labels, test_size=1-train_size, random_state=0, shuffle = True, stratify = class_labels)
    
    #summary dict
    sumary_dict = {
        'image':[],
        'partition':[],
        'resolution':[],
        'label' : []                
    }
    
    for j in range(len(index_train)):
        i = index_train[j]
       
        if partition[i] == 'positive':
            image_abs_path = os.path.join(bimcv_datasets_posi_part, subject[i], session[i], 'mod-rx',  images_names[i])  
        else:
            image_abs_path = os.path.join(bimcv_datasets_neg_part, subject[i], session[i], 'mod-rx',  images_names[i])

        img_loaded = img.load_image(image_abs_path, 'gray') 
               

        if isinstance(img_loaded, np.ndarray):           
            image_enhanced = img.enhance_image(img_loaded, imshape)
            image_save_path = os.path.join(build_folder_path,'train', class_labels[i])

            if os.path.exists(image_save_path) == False:
                os.mkdir(image_save_path)

            success = img.save_image(os.path.join(image_save_path, images_names[i]), image_enhanced) 

            if success:                
                sumary_dict['image'].append(images_names[i]) #add image name to dict
                sumary_dict['partition'].append('train') #add image name to dict
                sumary_dict['resolution'].append(img_resolution)
                sumary_dict['label'].append(class_labels[i])
    
    
    for j in range(len(index_vt) ):
        i = index_vt[j]       
       
        if partition[i] == 'positive':
            image_abs_path = os.path.join(bimcv_datasets_posi_part, subject[i], session[i], 'mod-rx', images_names[i])  
        else:
            image_abs_path = os.path.join(bimcv_datasets_neg_part, subject[i], session[i], 'mod-rx', images_names[i])

        img_loaded = img.load_image(image_abs_path, 'gray') 
        if isinstance(img_loaded, np.ndarray):           
            image_enhanced = img.enhance_image(img_loaded, imshape)
            if j < int(0.5 * len(index_vt)):                
                image_save_path = os.path.join(build_folder_path,'test', class_labels[i]) 
                part_str = 'test'
            else:                
                image_save_path = os.path.join(build_folder_path,'validation', class_labels[i])  
                part_str = 'validation'

            if os.path.exists(image_save_path) == False:
                os.mkdir(image_save_path)

            success = img.save_image(os.path.join(image_save_path, images_names[i]), image_enhanced)

            if success:                
                sumary_dict['image'].append(images_names[i]) #add image name to dict
                sumary_dict['partition'].append(part_str) #add image name to dict
                sumary_dict['resolution'].append(img_resolution)
                sumary_dict['label'].append(class_labels[i])
    
    #saving tsv    
    sumary_dict_df = pd.DataFrame(sumary_dict) 
    sumary_dict_df.to_csv(os.path.join(build_folder_path, 'bimcv_dataset.tsv'), '\t')
    master_cxr_df.to_csv(os.path.join(build_folder_path, 'master_bimcv_dataset.tsv'), '\t')      
            

except FileNotFoundError as e:
    print(e)
except KeyError as ke:
    print(ke)

