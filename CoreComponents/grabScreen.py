import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

from typing import Tuple

class ScreenGraber():
    def __init__(self) -> None:
        self.hwin = win32gui.GetDesktopWindow()
        self.hwindc = win32gui.GetWindowDC(self.hwin)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = win32ui.CreateBitmap()

    def grab_screen_RGBA(self, region: Tuple[int]=None) -> np.ndarray:

        if region:
                left,top,width,height = region
        else:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        self.bmp.CreateCompatibleBitmap(srcdc, width, height)
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (width, height), self.srcdc, (left, top), win32con.SRCCOPY)
    
        signedIntsArray = self.bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width,4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return img

    def grab_screen_RGB(self, region: Tuple[int]=None) -> np.ndarray:
        return cv2.cvtColor(grab_screen_RGBA(region), cv2.COLOR_RGBA2RGB)

    def grab_screen_GRAY(self, region: Tuple[int]=None) -> np.ndarray:
        return cv2.cvtColor(grab_screen_RGBA(region), cv2.COLOR_RGBA2GRAY)

def grab_screen_RGBA(region: Tuple[int]=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,width,height = region
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

def grab_screen_RGB(region=None):
    return cv2.cvtColor(grab_screen_RGBA(region), cv2.COLOR_RGBA2RGB)

def grab_screen_GRAY(region=None):
    return cv2.cvtColor(grab_screen_RGBA(region), cv2.COLOR_RGBA2GRAY)

if __name__ == "__main__":
    screen = grab_screen_RGBA(region=(0, 0, 1280, 720))
    print(screen.shape)