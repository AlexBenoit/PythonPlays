
import numpy as np
import os, os.path
import tensorflow as tf


def sorting_function(text):
    number = int(''.join(filter(str.isdigit, text)))
    return number

files_in_directory = [name for name in os.listdir('./Training Data') if os.path.isfile("./Training Data/" + name)]
files_in_directory.sort(key=sorting_function)
files_in_directory = np.array(files_in_directory)
all_data_screens = []
all_data_inputs = []

for file in files_in_directory[:]:
    print("Loading started for : " + file)
    try:
        data = np.load('./Training Data/' + file)
        screens = data.f.arr_0.copy().tolist()
        inputs = data.f.arr_1.copy().tolist()
        for i, screen in enumerate(screens):
            all_data_screens.append(screen)
            all_data_inputs.append(inputs[i])
    except:
        print("An error occured on file: " + file)

print("saving file")
np.savez_compressed('output/data_jim_1', all_data_screens, all_data_inputs)