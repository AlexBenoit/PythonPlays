import cv2
import numpy as np
import pickle

class TextAnalyzer():

    def __init__(self):
     

        filename = 'finalized_number_model.sav'

        loaded_model = pickle.load(open(filename, 'rb'))

        self.mlp = loaded_model
        

    def predict(self, img):
        test_cells_smash = []

        for i in img:
            d = i.flatten()
            test_cells_smash.append(d)
        
        
        test_cells_smash = np.array(test_cells_smash,dtype=np.float32)
        predictions = self.mlp.predict(test_cells_smash)
       
        return  predictions
        
if __name__ == "__main__":
    digitAnalzer = TextAnalyzer()
    digitAnalzer.predict()