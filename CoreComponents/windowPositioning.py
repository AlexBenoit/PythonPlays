import win32gui
import time
import subprocess
import os
import ctypes




def positionWindow():#(handle, x, y, width, height):
    windowHandle = win32gui.FindWindow(None, 'Dolphin')
    awareness = ctypes.c_int()

    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    print('---------------------')
    print(awareness.value)
    print('---------------------')
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)


    win32gui.MoveWindow(windowHandle, -7, 0, 1280, 720, 1) #-7 = invisible border dans windows 10

def openWindow(): 

    pwd = os.getcwd()
    
    print(pwd)
    subprocess.Popen([pwd +'\..\Dolphin\Dolphin.exe', '-b', '-e=' + pwd + '\..\Dolphin\ISOs\Super Smash Bros. Melee (USA).iso']) #Hard coded path
    time.sleep(2) #laisse le temps a l'emulateur de launch
