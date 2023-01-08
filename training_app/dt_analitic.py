#IMPORTS*********************************************************************************
#standard lib
import configparser
import os


#other imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

#parsers
config = configparser.ConfigParser()

#Getting configuration ------------------------------------------------------------
config.read('m_cxr_generator.conf')

#path to bimcv dataset
bimcv_path = os.path.abspath(config['paths']['bimcv_covid19'])

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

    #Datasets Secction***********************************************************************
    master_dataset_df = pd.read_table(os.path.join(bimcv_path,'master_cxr_dataset.tsv'))
    #master_dataset_df = pd.read_table('master_cxr_dataset.tsv')
    #Master subdataset from negative partition
    master_negpart_df = master_dataset_df.loc[master_dataset_df['partition'].str.contains('negative', case=False)]
    master_negpart_positive_df = master_negpart_df.loc[master_negpart_df['covid_pneumonia'].str.contains('positive', case=False)]
    master_negpart_negative_df = master_negpart_df.loc[master_negpart_df['covid_pneumonia'].str.contains('negative', case=False)]
    #Master subdataset from positive partition
    master_posipart_df = master_dataset_df.loc[master_dataset_df['partition'].str.contains('positive', case=False)]
    master_posipart_positive_df = master_posipart_df.loc[master_posipart_df['covid_pneumonia'].str.contains('positive', case=False)]
    master_posipart_negative_df = master_posipart_df.loc[master_posipart_df['covid_pneumonia'].str.contains('negative', case=False)]

    #Totals Secction*************************************************************************
    images_totals = master_dataset_df.index[-1] + 1 
    #Patiens totals
    patiens_df = master_dataset_df.drop_duplicates(subset=['PatientID'])
    #Male patiens totals
    patiens_male = len(master_dataset_df.loc[master_dataset_df['patient_sex'].str.contains('M', case=False)] )
    #Female patiens totals
    patiens_female = len(master_dataset_df.loc[master_dataset_df['patient_sex'].str.contains('F', case=False)] )

    #Radiological findings for negative partition***************************************************************
    negpart_patient_positive_findings = {}

    for index, row in master_negpart_positive_df.iterrows(): 
        sub = row.loc['PatientID']
        ses = row.loc['ReportID']

        #getting all rows for subject sub
        subject_label_rows = dataset_neg_img_labels.loc[dataset_neg_img_labels['PatientID'].str.contains(sub, case=False)]
        #getting all session
        report_row = subject_label_rows.loc[subject_label_rows['ReportID'].str.contains(ses, case=False)]
        #get label value column
        patien_neg_cui = report_row['labelCUIS']
        patien_neg_cui_labels = patien_neg_cui.array[0][1:-1]
        patien_neg_cui_list = patien_neg_cui_labels.split(',')
        
        for cui in patien_neg_cui_labels.split(','):   
            if cui != '':         
                if cui in negpart_patient_positive_findings:                
                    negpart_patient_positive_findings[cui] += 1
                else:
                    negpart_patient_positive_findings[cui] = 1

   #Radiological findings for positive partition***************************************************************
    posipart_patient_positive_findings = {}

    for index, row in master_posipart_positive_df.iterrows(): 
        sub = row.loc['PatientID']
        ses = row.loc['ReportID']

        #getting all rows for subject sub
        subject_label_rows = dataset_posi_img_labels.loc[dataset_posi_img_labels['PatientID'].str.contains(sub, case=False)]
        #getting all session
        report_row = subject_label_rows.loc[subject_label_rows['ReportID'].str.contains(ses, case=False)]
        #get label value column
        patien_posi_cui = report_row['labelCUIS']
        if len(patien_posi_cui.array) > 0:
            patien_posi_cui_labels = patien_posi_cui.array[0][1:-1]
        else:
            patien_posi_cui_labels = ''
        
        patien_posi_cui_list = patien_posi_cui_labels.split(',')
        
        for cui in patien_posi_cui_labels.split(','): 
            if cui != '':                    
                if cui in posipart_patient_positive_findings :                
                    posipart_patient_positive_findings[cui] += 1
                else:
                    posipart_patient_positive_findings[cui] = 1

    #Ploting ***********************************************************************************************
    cuis = posipart_patient_positive_findings.keys()
    columns = [key for key in cuis if posipart_patient_positive_findings[key] >= 100]
    data = [posipart_patient_positive_findings[val] for val in columns]
    index = np.arange(len(columns)) + 1.3
    bar_width = 0.6
    plt.bar(index, data, bar_width)
    plt.xticks(index, columns, rotation='vertical')
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('CUI count')
    plt.show()
    
except FileNotFoundError as e:
    print(e)
except KeyError as ke:
    print(ke)