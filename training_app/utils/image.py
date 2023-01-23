import numpy as np
import cv2

def save_image(name, img):
    return cv2.imwrite(name, img)

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
    #resizing image
    enhimage = cv2.resize(img, shape)   
    #normalizing image
    enhimage = normalize_image(enhimage, max_range) 
    # aplying clahe algoritms
    enhimage = clahe_filtering(enhimage)
    #returning an images with values between 0-1 instead off 0-255
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
    
if __name__ == '__main__':
    #replacing path for images
    image_width = 800
    image_height = 800
    imshape = (image_width, image_height)

    #loading test image

    try:
        image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

        rezised_img = cv2.resize(image, imshape)        
        cv2.imshow('original', rezised_img)

        norm_img = normalize_image(rezised_img, 200)
        cv2.imshow('normalized', norm_img)

        clahe_img = clahe_filtering(norm_img)
        cv2.imshow('filtered', clahe_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except (FileNotFoundError):
        print('test.png image not found')