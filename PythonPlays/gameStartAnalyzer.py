import cv2
import numpy as np
from PIL import Image
import glob
import scipy.io as spio
import time
from sklearn.neural_network import MLPClassifier
import pickle

class GameStartAnalyzer:

    def __init__(self):

   
        filename = 'finalized_gamestartDetector_model.sav'

        loaded_model = pickle.load(open(filename, 'rb'))

        self.mlp = loaded_model
        

    def predict(self, img):
        test_cells_smash = []

        d = img.flatten()
        test_cells_smash.append(d)
        
        
        test_cells_smash = np.array(test_cells_smash,dtype=np.float32)
        predictions = self.mlp.predict(test_cells_smash)
       
        return  predictions[0]
        
if __name__ == "__main__":
    digitAnalzer = GameStartAnalyzer()