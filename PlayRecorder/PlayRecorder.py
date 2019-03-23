#!/usr/bin/python

#External imports
import keyboard
import csv
import time


#Internal imports
import windowPositioning
import Recorder
import Writer

#Specific imports
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT

recording = False

def start_recording():
    ready_to_record = False

    while not ready_to_record:
        if keyboard.is_pressed("r"):
            ready_to_record = True

    recording = True
    index = 0

    while recording == True:
        if  keyboard.is_pressed("esc"):
           recording = False
        data = record(index)
        new_writer = Writer.Writer(index, data)
        new_writer.start()
        index = index + 1
       
    
def record(index):
    record_thread = Recorder.Recorder(index)
    record_thread.start()
    record_thread.join()
    return record_thread.get_data()



def main():
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)

    window = windowPositioning.openWindow("Smash Melee")
    window.positionWindow(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    start_recording()

if __name__ == '__main__':
    main()