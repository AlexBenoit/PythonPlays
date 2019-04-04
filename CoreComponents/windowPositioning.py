import win32gui
import time
import subprocess
import os
import ctypes
import win32con

class Window:
    def __init__(self, windowHandle):
        self.windowHandle = windowHandle
    
    def positionWindow(self, x, y, width, height):
        lStyle = win32gui.GetWindowLong(self.windowHandle, win32con.GWL_STYLE)
        lExStyle = win32gui.GetWindowLong(self.windowHandle, win32con.GWL_EXSTYLE)
        lStyle &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | win32con.WS_SYSMENU)
        lExStyle &= ~(win32con.WS_EX_DLGMODALFRAME | win32con.WS_EX_CLIENTEDGE | win32con.WS_EX_STATICEDGE)
        win32gui.SetWindowLong(self.windowHandle, win32con.GWL_STYLE, lStyle);
        win32gui.SetWindowLong(self.windowHandle, win32con.GWL_EXSTYLE, lExStyle);

        win32gui.SetWindowPos(self.windowHandle, 0, x, y, width, height, 0x0040)

def openWindow(window): 
    pwd = os.getcwd()
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(1)

    if(window == "Smash Melee"):
        dolphin = subprocess.Popen([pwd +'\..\Dolphin\Dolphin.exe', '-b', '-e=' + pwd + '\..\Dolphin\ISOs\Super Smash Bros. Melee (USA).iso']) #Hard coded path
        time.sleep(2) #laisse le temps a l'emulateur de launch
        return Window(win32gui.FindWindow(None, 'Dolphin'))