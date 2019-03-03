import win32gui
import time
import subprocess
import os
import ctypes

class Window:
    def __init__(self, windowHandle):
        self.windowHandle = windowHandle
    
    def positionWindow(self, x, y, width, height):
        win32gui.SetWindowPos(self.windowHandle, 0, x - 8, y, width + 16, height + 8, 0x0040) #-7 = invisible border dans windows 10

def openWindow(window): 
    pwd = os.getcwd()
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)

    if(window == "Smash Melee"):
        dolphin = subprocess.Popen([pwd +'\..\Dolphin\Dolphin.exe', '-b', '-e=' + pwd + '\..\Dolphin\ISOs\Super Smash Bros. Melee (USA).iso']) #Hard coded path
        time.sleep(2) #laisse le temps a l'emulateur de launch
        print(dolphin.pid)
        return Window(win32gui.FindWindow(None, 'Dolphin'))