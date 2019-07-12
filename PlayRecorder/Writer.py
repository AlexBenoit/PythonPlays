from threading import Thread
import numpy as np
import time


class Writer(Thread):

    def __init__(self, index, data_screens, data_inputs):
        Thread.__init__(self)
        self.index = index
        self.data_screens = data_screens
        self.data_inputs = data_inputs

    def run(self):
        start = time.time()
        print("Saving Started [" + str(self.index) + "]")
        np.savez_compressed('output/data_' + str(self.index), self.data_screens, self.data_inputs)
        end = time.time()
        print("Save Finished [" + str(self.index) + "] (" + str(end-start) + " seconds)")