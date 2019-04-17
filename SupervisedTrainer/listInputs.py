
#External imports
import json
import os, os.path
import numpy as np
import time

def main():
    index_string = input("What index should we start training ? ")
    index = int(index_string)
    files_in_directory = [name for name in os.listdir('./Training Data') if os.path.isfile("./Training Data/" + name)]
    files_in_directory = files_in_directory[index:]

    found_inputs = []
    for file in files_in_directory:
        start = time.time()
        print("Training started for : " + file)
        try:
            data = np.load('./Training Data/' + file)
            data_inputs = data.f.arr_1.copy()
            for i, input_1 in enumerate(data_inputs):

                print(input_1)
                found_in_array = False

                for current_input in found_inputs:
                    if np.array_equal(input_1, current_input):
                        found_in_array = True
                    
                if not found_in_array:
                    found_inputs.append(list(input_1))
                    print("Added")

            end = time.time()
            print("Time for file : " + file + " = " + str(end-start) + " seconds")
          
        except:
            print("Can not load file")
        
    print(found_inputs)
    
    with open('../list_inputs.json', 'r') as infile:
        current_data_inputs = json.load(infile)

    for current_input in found_inputs:
        found_in_array = False

        for current_input2 in current_data_inputs:
            if np.array_equal(current_input, current_input2):
                found_in_array = True
                break

        if not found_in_array:
            current_data_inputs.append(list(current_input))
            print("Added")
    print(current_data_inputs)
    with open('../list_inputs.json', 'w') as outfile:
        json.dump(current_data_inputs, outfile)

if __name__ == "__main__":
    main()