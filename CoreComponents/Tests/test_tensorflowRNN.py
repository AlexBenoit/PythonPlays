import unittest
import numpy as np
import cv2
import json

from tensorflowRNN import RNNAgent
from globalConstants import RECORDING_WIDTH, RECORDING_HEIGHT

class Test_tensorflowRNN(unittest.TestCase):
    def test_prediction(self):
        list_inputs = None
        with open('../list_inputs.json', 'r') as infile:
            list_inputs = json.load(infile)

        mock_screen = np.array(np.random.randint(0,255, RECORDING_WIDTH * RECORDING_HEIGHT), dtype="uint8")
        mock_screen = mock_screen.reshape((RECORDING_HEIGHT, RECORDING_WIDTH))
        self.assertEqual(mock_screen.shape, (RECORDING_HEIGHT, RECORDING_WIDTH))
        self.assertTrue((mock_screen >= 0).all() and (mock_screen <= 255).all())

        solver = RNNAgent(RECORDING_HEIGHT*RECORDING_WIDTH, len(list_inputs))
        action = solver.get_action(mock_screen)
        self.assertEqual(len(action), len(list_inputs))
        self.assertTrue((action >= -1).all() and (action <= 1).all())

if __name__ == '__main__':
    unittest.main()
