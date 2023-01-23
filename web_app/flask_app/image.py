import cv2
import numpy as np



def load_image(img_file):

    try:  
        
        img_str = img_file.read()  
        img_np = np.fromstring(img_str, np.uint8)       
        img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)       
        
        return img
        
    except:
        print('Error loading image: ', img_file)
        return None

def enhance_image(img, shape):
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
    refimage = np.zeros(img.shape)          #reference image for normalization
    #normalizing image
    image = cv2.normalize(img, refimage, 0, 255, cv2.NORM_MINMAX)

    # aplying clahe algoritms
    clahe = cv2.createCLAHE(clipLimit = 10) #clahe filter   
    # aplying clahe algoritms
    filter_image =  clahe.apply(image)
    #resizing image
    enhimage = cv2.resize(filter_image, shape)
    #return enhimage/255.0
    return enhimage

def show(img):
    print('image from server')
    cv2.imshow('uploaded', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()