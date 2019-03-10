#!/usr/bin/python

# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

ESCAPE = 0x01
NUM_1 = 0x02
NUM_2 = 0x03
NUM_3 = 0x04
NUM_4 = 0x05
Q = 0x10
W = 0x11
E = 0x12
R = 0x13
T = 0x14
Y = 0x15
U = 0x16
I = 0x17
O = 0x18
P = 0x19
RETURN = 0x1C
L_CONTROL = 0x1D
A = 0x1E
S = 0x1F
D = 0x20
F = 0x21
G = 0x22
H = 0x23
J = 0x24
K = 0x25
L = 0x26
L_SHIFT = 0x2A
Z = 0x2C
X = 0x2D
C = 0x2E
V = 0x2F
B = 0x30
N = 0x31
M = 0x32
L_ALT = 0x38
SPACE = 0x39
ARROW_UP = 0xC8
ARROW_LEFT = 0xCB
ARROW_RIGHT = 0xCD
ARROW_DOWN = 0xD0
#LMB =
#RMB =
#MMB =
#MMB =

inputDist = {
    "x" : [0,X],
    "z" : [1,Z], 
    "c" : [2,C], 
    "s" : [3,S], 
    "d" : [4,D], 
    "up" : [5,ARROW_UP], 
    "down" : [6,ARROW_DOWN], 
    "left" : [7,ARROW_LEFT], 
    "right" : [8,ARROW_RIGHT], 
    "i" : [9,I],   
    "k" : [10,K],     
    "j" : [11,J],     
    "l" : [12,L],      
    "q" : [13,Q], 
    "w" : [14,W]
}

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions
def pressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def releaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def mouseLeftClick():
    ctypes.windll.user32.GetCursorInfo()
    ctypes.windll.user32.SetCursorPos(100, 20)
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0) # left down
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0) # left up
    
if __name__ == '__main__':
    PressKey(0x11)
    time.sleep(1)
    ReleaseKey(0x11)
    time.sleep(1)
    ctypes.windll.user32.SetCursorPos(0,0)
    print(ctypes.windll.user32.GetCursorPos())
