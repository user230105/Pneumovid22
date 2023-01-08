#IMPORTS*********************************************************************************
#standard lib
import configparser
from pickle import TRUE

#other imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import metrics

def save_eval_results(performance, classes, file_path):
    

    evalparser = configparser.ConfigParser()
    evalparser['Classes'] = classes
    evalparser['performance'] = {}
    evalparser['performance']['loss'] = str(performance[0])
    evalparser['performance']['accuracy'] = str(performance[1])    

    with open(file_path, 'w') as eval_file:
        evalparser.write(eval_file)

def plot_training(history, cat = 'loss'):
    # summarize history for loss
    plt.plot(history.history[cat])  
    plt.plot(history.history['val_'+ cat])  
    plt.title('model ' + cat)
    plt.ylabel(cat)
    plt.xlabel('epoch')   
    plt.legend(['train', 'validation'], loc='upper left') 
    plt.show()

def binary_eval(y_p_multiclass, y_t_multiclass, classes):
    ''' function to eval with binary metrics multiclass model

    Parameters:
    -----------
    y_p_multiclass: np.ndarray
        tensor with model's predictions
    y_t_multiclass: np.ndarray
        tensor with true labels
    classes: list of model classes
        
    Returns:
    -------
     metrics: Tuple
       (Binary_acurcy, precision, recall)
    ''' 

    if y_p_multiclass.shape != y_t_multiclass.shape:
        print('Binary eval Error, labels tensors must have the same shape')
        print('y_true: ', y_t_multiclass.shape)
        print('y_predict: ', y_p_multiclass.shape)
        return(0, 0, 0)

    #metrics
    m_bin_acc = metrics.BinaryAccuracy()
    m_precision = metrics.Precision()
    m_recall = metrics.Recall()
    
    #get covid class index
    covid_index = np.argmax(classes == 'covid') 
    print('index of covid class', covid_index)
    #seting a [1] for positive and [0] for negatives
    covid_positive = np.ones(1)

    y_p = np.zeros((y_p_multiclass.shape[0], 1))
    y_p_class = np.argmax(y_p_multiclass, 1)
    y_p_posi_mask = y_p_class == covid_index
    y_p[y_p_posi_mask] = covid_positive    
    

    y_t = np.zeros((y_t_multiclass.shape[0], 1))
    y_t_class = np.argmax(y_t_multiclass, 1)
    y_t_posi_mask = y_t_class == covid_index    
    y_t[y_t_posi_mask] = covid_positive

    #metrics calculation
    m_bin_acc.update_state(y_t, y_p)
    m_precision.update_state(y_t, y_p)
    m_recall.update_state(y_t, y_p)


    binary_accuracy = m_bin_acc.result().numpy()
    precision = m_precision.result().numpy()
    recall = m_recall.result().numpy()

    #print('y_t',y_t_posi_mask)
    #print('y_p',y_p_posi_mask)
    #print('metrics ', binary_accuracy, precision, recall)
    return (binary_accuracy, precision, recall)
    

if __name__ == '__main__':
    print('starting test')
    a = [0.2, 0.8]
    b= [0.6, 0.2]
    test = np.array([a, a, b, a, b, b, b, b, a, b, a, a, a, b, a, a, b, a])
    print('test:', test)
    y_t = np.argmax(test, 1)
    print('y_t', y_t)
    mask = y_t == 0
    print(mask)
