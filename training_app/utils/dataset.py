import pandas as pd
import os
import re

def print_dt(dict):
    for i in range(len(dict['image'])):
        print("Image: %s, partition: %s , label:%s" %(dict['image'][i], dict['partition'][i], dict['class'][i]))

def create_folder_struct(build_folder_path):
    #creating dataset root folder
    if os.path.exists( build_folder_path) == False:  
        os.mkdir(build_folder_path)
    #creating train folder
    if os.path.exists( os.path.join(build_folder_path, 'train')) == False:  
        os.mkdir(os.path.join(build_folder_path, 'train'))
    #creating test folder
    if os.path.exists( os.path.join(build_folder_path, 'test')) == False:  
        os.mkdir(os.path.join(build_folder_path, 'test'))
    #creating test folder
    if os.path.exists( os.path.join(build_folder_path, 'validation')) == False:  
        os.mkdir(os.path.join(build_folder_path, 'validation'))

def merge_dict(dict1, dict2):
    ''' function to merge two dataset dict

    Parameters:
    -----------
    dict1: Dictionary
        dataset dictionary
    dict2: Dictionary
        dataset dictionary
    Returns:
    -------
     dict: Dictionary
       a merged dataset dictonary 
    ''' 
    for i in range(len(dict2['image'])):
        
        if (dict2['image'][i] in dict1['image']) == False:            
            dict1['image'].append(dict2['image'][i])
            dict1['partition'].append(dict2['partition'][i])
            dict1['PatientID'].append(dict2['PatientID'][i])
            dict1['ReportID'].append(dict2['ReportID'][i])
            dict1['patient_age'].append(dict2['patient_age'][i])
            dict1['patient_sex'].append(dict2['patient_sex'][i])
            dict1['class'].append(dict2['class'][i])

    return dict1

def discard_dict_entry(dict):
    ''' function to delete all entry with label discard

    Parameters:
    -----------
    dict: Dictionary
        dataset dictionary
    

    Returns:
    -------
     dict: Dictionary
       a dataset dictonary with class label field
    ''' 

    #new lists
    image = []
    partition = []
    PatientID = []
    ReportID = []
    patient_age = []
    patient_sex = []
    label = []

    #eliminating keys if discard label is present
    for i in range(len(dict['image'])):
        if dict['class'][i] != 'discard':
            image.append(dict['image'][i])
            partition.append(dict['partition'][i])
            PatientID.append(dict['PatientID'][i])
            ReportID.append(dict['ReportID'][i])
            patient_age.append(dict['patient_age'][i])
            patient_sex.append(dict['patient_sex'][i])
            label.append(dict['class'][i])
    
    dict['image'] = image
    dict['partition'] = partition
    dict['PatientID'] = PatientID
    dict['ReportID'] = ReportID
    dict['patient_age'] = patient_age
    dict['patient_sex'] = patient_sex
    dict['class'] = label

    return dict

def get_labels_multiclass(sub, ses, labels_df, partition):
    ''' function to get labels for images

    Parameters:
    -----------
    sub: string
        subject
    ses: string
        session
    labels_df: pandas.Dataframe
        dataframe with labels and radiological findings
    test_df: pandas.Dataframe
        result of covid test

    Returns:
    -------
     class: string
       covid normal, or nocovid labels class
    ''' 
            
    #getting all rows for subject sub
    subject_label_rows = labels_df.loc[labels_df['PatientID'].str.contains(sub, case=False)]           
    
    #getting all session
    report_row = subject_label_rows.loc[subject_label_rows['ReportID'].str.contains(ses, case=False)]

    #get label cell value from column labelCUIS
    covid_label_cell = report_row['labelCUIS']
    if len(covid_label_cell.array) > 0:
        covid_label_string = covid_label_cell.array[0][1:-1]
    else:
        covid_label_string = ''

    if covid_label_string.find('C0205307') != -1:
        #if normal CUI exits on string        
        return 'normal'
    elif covid_label_string.find('C5203670') != -1:
        #if covid 19  diagnosis tag is present
        if partition == 'positive':                            
            return 'covid_pneumonia'            
        else: 
            return 'discard'        
    elif (covid_label_string.find('C0032285') != -1 or covid_label_string.find('C0032310') != -1 or covid_label_string.find('C5203671') != -1):
        #if pneumonia or viral pneumonia or uncertain of covid check for test result
        if partition == 'positive':                           
            return 'covid_pneumonia'           
        else:
            #descarto las imagenes confusas, aquellas que pueden ser covid pero no estan confirmadas            
            return 'discard'
    elif (covid_label_string.find('C0277877') != -1 or
            covid_label_string.find('C2073538') != -1 or
            covid_label_string.find('C3544344') != -1 or 
            covid_label_string.find('C2073672') != -1 or 
            covid_label_string.find('C2073583') != -1):
        #if there are ground glass pattern or consolidations or also infiltration
        if partition == 'positive':            
                return 'covid_pneumonia'            
        else:
            #si son de la particion negativa las clasifico como abnormal o sea no normal
            return 'abnormal'  
    elif (covid_label_string.find('C1332240') != -1 or            
            covid_label_string.find('C0521530') != -1):
        #if there are ground glass pattern or consolidations or also infiltration
        if partition == 'positive':            
                return 'covid_pneumonia'            
        else:
            #si son de la particion negativa las clasifico como abnormal o sea no normal
            return 'abnormal'    
    else:
        if covid_label_string == '':
            return 'normal'
        if partition == 'positive':           
            return 'discard'           
        else:            
            return 'abnormal'

def get_labels_binary(sub, ses, labels_df, partition):
    ''' function to get labels for images

    Parameters:
    -----------
    sub: string
        subject
    ses: string
        session
    labels_df: pandas.Dataframe
        dataframe with labels and radiological findings
    test_df: pandas.Dataframe
        result of covid test

    Returns:
    -------
     class: string
       positive or negative label
    ''' 
            
    #getting all rows for subject sub
    subject_label_rows = labels_df.loc[labels_df['PatientID'].str.contains(sub, case=False)]           
    
    #getting all session
    report_row = subject_label_rows.loc[subject_label_rows['ReportID'].str.contains(ses, case=False)]

    #get label cell value from column labelCUIS
    covid_label_cell = report_row['labelCUIS']
    if len(covid_label_cell.array) > 0:
        covid_label_string = covid_label_cell.array[0][1:-1]
    else:
        covid_label_string = ''

    if covid_label_string.find('C0205307') != -1:
        #if normal CUI exits on string        
        return 'negative'
    elif covid_label_string.find('C5203670') != -1:
        #if covid 19  diagnosis tag is present
        if partition == 'positive':                            
            return 'positive'            
        else: 
            return 'discard'        
    elif (covid_label_string.find('C0032285') != -1 or covid_label_string.find('C0032310') != -1 or covid_label_string.find('C5203671') != -1):
        #if pneumonia or viral pneumonia or uncertain of covid check for test result
        if partition == 'positive':                           
            return 'positive'           
        else:
            #descarto las imagenes confusas, aquellas que pueden ser covid pero no estan confirmadas            
            return 'discard'
    elif (covid_label_string.find('C0277877') != -1 or
            covid_label_string.find('C2073538') != -1 or
            covid_label_string.find('C3544344') != -1 or 
            covid_label_string.find('C2073672') != -1 or 
            covid_label_string.find('C2073583') != -1):
        #if there are ground glass pattern or consolidations or also infiltration
        if partition == 'positive':            
                return 'positive'            
        else:
            #si son de la particion negativa las clasifico como abnormal o sea no normal
            return 'negative'  
    elif (covid_label_string.find('C1332240') != -1 or            
            covid_label_string.find('C0521530') != -1):
        #if there are ground glass pattern or consolidations or also infiltration
        if partition == 'positive':            
                return 'positive'            
        else:
            #si son de la particion negativa las clasifico como abnormal o sea no normal
            return 'negative'    
    else:
        if covid_label_string == '':
            return 'negative'
        if partition == 'positive':           
            return 'discard'           
        else:            
            return 'negative'

def get_dicom_age(age_str):

    ''' function to parse the dicom's age string   

    Parameters:
    -----------
    age_str: string
        dicom age string

    Returns:
    -------
     age: number
       number of years for valid string otherwise 0
    ''' 
    year_flag_index = age_str.find('Y')
    if year_flag_index == -1:
        return 0
    else:
        return int(age_str[0:year_flag_index])

def get_session_info(session_info):
    
    ''' function to parse the session info scans.tsv file  

    Parameters:
    -----------
    session_info: string
        scans.tsv's name

    Returns:
    -------
     sesion_info: dict
        dictionary containing {image:[valid images], patient_age: number, patient_sex : 'M|F'}
    ''' 
    #Regular expresion definition for filter patterns on strings ***********************
    #regular expresion for cxr images
    cxr_image_regex = re.compile('.*(CR|DX).*\.png', re.IGNORECASE) 
    #regular expresion for frontal chest x ray cxr
    frontal_cxr_regex = re.compile('.*(AP|PA).*', re.IGNORECASE)
    #*********************************************************************************

    session_info_dict = {
        'image':[]  
    }

    try:
        #opening with pandas
        session_info_df = pd.read_table(session_info)                            
        for index, row in session_info_df.iterrows(): 
        
            #filter cxr image. discarting ct
            if cxr_image_regex.match(row['filename']) != None and index == 0:    #only firs image on scan sesion file            
                #filter frontal radiography, discarting lateral
                if frontal_cxr_regex.match(row['filename']) != None or frontal_cxr_regex.match(row['Series Description (0008103E)']) != None:

                    #checks for double filenames. ejem image1,image2
                    comma_index = row['filename'].find(',')
                    if comma_index == -1:
                        #simple filename
                        session_info_dict['image'].append(row['filename'])                      
                        
                    else:
                        images = row['filename'].split(',')
                        session_info_dict['image'].append(images[0]) 

                    session_info_dict['patient_age'] = get_dicom_age(row["Patient's Age (00101010)"])
                    session_info_dict['patient_sex'] = (row["Patient's Sex (00100040)"])
        
        if len(session_info_dict['image']) > 0:
            return session_info_dict
        else:
            return None

    except FileNotFoundError:
        print('failing loading %s, file not found' % (session_info))
    except:
        pass

def scan_dataset(dataset_path, partition = 'positive'):

    ''' function to walk through all dataset subject's folder  

    Parameters:
    -----------
    dataset_path: string
        Path to dataset directory that contain subject's folders
    partition: string
        dataset partition to be scanned
    Returns
    -------
     dataset_info: Dict
    ''' 
    #Regular expresion definition for scans patterns on names***********************
    #regular expresion that match subject folder's name
    subject_regex = re.compile('sub-s[\d]*', re.IGNORECASE)
    #regular expresion that match sesion folder's name
    session_regex = re.compile('ses-e[\d]*', re.IGNORECASE)
    #regular expresion for scan tsv file
    scans_tsv_reges = re.compile('(sub-s[\d]*)?_(ses-e[\d]*)?_scans.tsv', re.IGNORECASE)    
    #*********************************************************************************

    #Defining subjects dict ******************************************************
    subjects_dict = {
        'image':[],
        'partition':[],
        'PatientID':[],
        'ReportID' : [],
        'patient_age':[],
        'patient_sex':[],        
    }
    #******************************************************************************

    try:
        dataset_basedir_content = os.scandir(dataset_path)

        #Walking through all subject folder an making master dataframe******************** 
        print('Scanning dataset ', dataset_path)   

        for direntry in dataset_basedir_content: 
            #checing that is a subject directory and subject name
            if direntry.is_dir() and subject_regex.match(direntry.name) != None: 
                subject = direntry.name #subject name   

                #getting sessions inside subject folder
                subject_dir_content = os.scandir(direntry.path)

                for subject_entry in subject_dir_content:
                    #checking that is a session directory and session name
                    if subject_entry.is_dir() and session_regex.match(subject_entry.name) != None:

                        session = subject_entry.name
                        session_dir_content = os.scandir(subject_entry.path)

                        for session_entry in session_dir_content:
                            #checking that is a session scans tsv file
                            if session_entry.is_file() and scans_tsv_reges.match(session_entry.name) != None:
                                #opening with pandas
                                session_info_dict = get_session_info(session_entry.path) 

                                if session_info_dict != None:
                                    #add entry to subject dict
                                    for img in session_info_dict['image']:
                                        #checks for images with mod-rx in their name
                                        slash_index = img.find('/')
                                        if slash_index == -1:
                                            subjects_dict['image'].append(img)
                                        else:
                                            img_split_name = img[slash_index+1:]
                                            subjects_dict['image'].append(img_split_name)
                                        subjects_dict['partition'].append(partition)                                    
                                        subjects_dict['PatientID'].append(subject)
                                        subjects_dict['ReportID'].append(session)
                                        subjects_dict['patient_age'].append(session_info_dict['patient_age'])
                                        subjects_dict['patient_sex'].append(session_info_dict['patient_sex'])

        return subjects_dict                     

    except:
        print("Fail to scan subject's folder: ", dataset_path)
    pass
    
def scan_covidx_dataset(dataset_path, partition = 'train'):

    ''' function to walk through all partition image folder  

    Parameters:
    -----------
    dataset_path: string
        Path to dataset directory that contain image's folders
    partition: string
        dataset partition to be scanned
    Returns
    -------
     dataset_info: Dict
    '''

    #Defining subjects dict ******************************************************
    subjects_dict = {
        'images':[]               
    }

    try:
        dataset_basedir_content = os.scandir(dataset_path)

        #Walking through all subject folder an making master dataframe******************** 
        print('Scanning dataset ', dataset_path)   

        for direntry in dataset_basedir_content: 
            #checing that is a subject directory and subject name            
            if direntry.is_file() and direntry.name.lower().endswith(('.png', '.jpg', '.jpeg')): 
                subjects_dict['images'].append(direntry.name)                       


        return subjects_dict                     

    except Exception as e:
        print("Fail to scan  folder: ", dataset_path)
        print(e)
    pass