import unittest

import grabScreen

from globalConstants import RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT

class Test_grabScreen(unittest.TestCase):
    def test_grab_screen_gray(self):
        screen = grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
        self.assertEqual(screen.shape, (RECORDING_HEIGHT, RECORDING_WIDTH))
        self.assertTrue((screen >= 0).all() and (screen <= 255).all())
    
    def test_grab_screen_rgb(self):
        screen = grabScreen.grab_screen_RGB(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
        self.assertEqual(screen.shape, (RECORDING_HEIGHT, RECORDING_WIDTH, 3))
        self.assertTrue((screen >= 0).all() and (screen <= 255).all())

    def test_grab_screen_rgba(self):
        screen = grabScreen.grab_screen_RGBA(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
        self.assertEqual(screen.shape, (RECORDING_HEIGHT, RECORDING_WIDTH, 4))
        self.assertTrue((screen >= 0).all() and (screen <= 255).all())

if __name__ == '__main__':
    unittest.main()
