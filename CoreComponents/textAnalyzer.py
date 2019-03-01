import cv2
import numpy as np
from PIL import Image
import glob
import scipy.io as spio
import time
from sklearn.neural_network import MLPClassifier

class TextAnalyzer:
       
   

    def __init__(self):
        #load sample and train 
        image_list=[]
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*0/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*1/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*2/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*3/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*4/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*5/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*6/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*7/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*8/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)
        for filename in glob.glob('C:/Users/Nicolas/PFE/PythonPlays/CoreComponents/assets/*9/*.jpg'): 
            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            image_list.append(img)

        training_cells_smash = []
        for d in image_list:
            d = d.flatten()
            training_cells_smash.append(d)

        training_cells_smash = np.array(training_cells_smash,dtype=np.float32)

        cells_labels2 = []
        for i in range (0,90):
            cells_labels2.append(0)
        for i in range (0,98):
            cells_labels2.append(1)
        for i in range (0,9):
            cells_labels2.append(2)
        for i in range (0,17):
            cells_labels2.append(3)
        for i in range (0,14):
            cells_labels2.append(4)
        for i in range (0,12):
            cells_labels2.append(5)
        for i in range (0,16):
            cells_labels2.append(6)
        for i in range (0,11):
            cells_labels2.append(7)
        for i in range (0,6):
            cells_labels2.append(8)
        for i in range (0,10):
            cells_labels2.append(9)

        cells_labels = np.array(cells_labels2)

        mlp = MLPClassifier(hidden_layer_sizes=(200, 300, 100), max_iter=1000)
        print(len(training_cells_smash))
        print(len(cells_labels))
        mlp.fit(training_cells_smash, cells_labels)


        self.mlp = mlp
        

    def predict(self, img):
        test_cells_smash = []
        
        d = img.flatten()
        test_cells_smash.append(d)
        test_cells_smash = np.array(test_cells_smash,dtype=np.float32)
        predictions = self.mlp.predict(test_cells_smash)
       
        return  predictions[0]
        
if __name__ == "__main__":
    digitAnalzer = TextAnalyzer()
    digitAnalzer.predict()