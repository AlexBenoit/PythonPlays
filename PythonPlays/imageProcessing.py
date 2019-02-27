import cv2
import skimage
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

def process_img(image):
    original_image = image

    #Trouve les pourcentage ici
    pourcentage1 = None
    pourcentage2 = None

    return pourcentage1, pourcentage2



if __name__ == '__main__':
   print('fuck')
   clf = joblib.load("digits_cls.pkl")
   im = cv2.imread("photo_2.jpg")
   cv2.imshow("Resulting Image with Rectangular ROIs", im)