import win32gui
import time
import subprocess
import os
import ctypes
import ctypes.wintypes
import win32con
import sys   

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

    if(window == "Smash Melee"):
        dolphin = subprocess.Popen([pwd +'\..\Dolphin\Dolphin.exe', '-b', '-e=' + pwd + '\..\Dolphin\ISOs\Super Smash Bros. Melee (USA).iso']) #Hard coded path
        time.sleep(2) #laisse le temps a l'emulateur de launch
        return Window(win32gui.FindWindow(None, 'Dolphin'))
    elif (window == "For Honor"):
        print(pwd)
        for_honor = subprocess.Popen(["D:/Jeux Uplay/ForHonor/forhonor.exe"]) #Hard coded path
        return Window(wait_for_window_open("For Honor"))

def wait_for_window_open(window_name):
    window_found = False
    window_handle = None

    EVENT_SYSTEM_DIALOGSTART = 0x0010
    EVENT_SYSTEM_FOREGROUND = 0x0003
    WINEVENT_OUTOFCONTEXT = 0x0000

    user32 = ctypes.windll.user32
    ole32 = ctypes.windll.ole32

    ole32.CoInitialize(0)

    WinEventProcType = ctypes.WINFUNCTYPE(
        None, 
        ctypes.wintypes.HANDLE,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LONG,
        ctypes.wintypes.LONG,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.DWORD
    )

    def callback(hWinEventHook, event, hwnd, idObject, idChild, dwEventThread, dwmsEventTime):
        length = user32.GetWindowTextLengthA(hwnd)
        buff = ctypes.create_string_buffer(length + 1)
        user32.GetWindowTextA(hwnd, buff, length + 1)
        print(buff.value)
        if window_name in str(buff.value):
            print("Found the right window")
            window_found = True
            window_handle = hwnd

    WinEventProc = WinEventProcType(callback)

    user32.SetWinEventHook.restype = ctypes.wintypes.HANDLE
    hook = user32.SetWinEventHook(
        EVENT_SYSTEM_FOREGROUND,
        EVENT_SYSTEM_FOREGROUND,
        0,
        WinEventProc,
        0,
        0,
        WINEVENT_OUTOFCONTEXT
    )
    if hook == 0:
        print('SetWinEventHook failed')
        sys.exit(1)

    msg = ctypes.wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
        user32.TranslateMessageW(msg)
        user32.DispatchMessageW(msg)
        if window_found:
            break

    user32.UnhookWinEvent(hook)
    ole32.CoUninitialize()

    return window_handle