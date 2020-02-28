import unittest
import numpy as np
import cv2
import json

from tensorflowRNN import RNNAgent
from globalConstants import RECORDING_WIDTH, RECORDING_HEIGHT

class Test_tensorflowRNN(unittest.TestCase):
    def test_get_action(self):
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


    def test_remember(self):
        list_inputs = None
        with open('../list_inputs.json', 'r') as infile:
            list_inputs = json.load(infile)

        mock_screen1 = np.array(np.random.randint(0,255, RECORDING_WIDTH * RECORDING_HEIGHT), dtype="uint8")
        mock_screen1 = mock_screen1.reshape((RECORDING_HEIGHT, RECORDING_WIDTH))
        mock_screen2 = np.array(np.random.randint(0,255, RECORDING_WIDTH * RECORDING_HEIGHT), dtype="uint8")
        mock_screen2 = mock_screen2.reshape((RECORDING_HEIGHT, RECORDING_WIDTH))
        mock_screen3 = np.array(np.random.randint(0,255, RECORDING_WIDTH * RECORDING_HEIGHT), dtype="uint8")
        mock_screen3 = mock_screen3.reshape((RECORDING_HEIGHT, RECORDING_WIDTH))
        mock_screen4 = np.array(np.random.randint(0,255, RECORDING_WIDTH * RECORDING_HEIGHT), dtype="uint8")
        mock_screen4 = mock_screen4.reshape((RECORDING_HEIGHT, RECORDING_WIDTH))

        reward1 = 1
        reward2 = 2
        reward3 = 3

        solver = RNNAgent(RECORDING_HEIGHT*RECORDING_WIDTH, len(list_inputs))

        action1 = solver.get_action(mock_screen1)
        action2 = solver.get_action(mock_screen2)
        action3 = solver.get_action(mock_screen3)

        solver.remember(mock_screen1, action1, reward1, mock_screen2)
        solver.remember(mock_screen2, action2, reward2, mock_screen3)
        solver.remember(mock_screen3, action3, reward3, mock_screen4)

        self.assertEqual(3, len(solver.batch_old_screen))
        self.assertEqual(3, len(solver.batch_action))
        self.assertEqual(3, len(solver.batch_reward))
        self.assertEqual(3, len(solver.batch_screen))

        self.assertEqual(len(solver.batch_old_screen[0]), RECORDING_HEIGHT*RECORDING_WIDTH)

    def test_experience_replay(self):
        list_inputs = None
        with open('../list_inputs.json', 'r') as infile:
            list_inputs = json.load(infile)

        mock_screen1 = np.array(np.random.randint(0,255, RECORDING_WIDTH * RECORDING_HEIGHT), dtype="uint8")
        mock_screen1 = mock_screen1.reshape((RECORDING_HEIGHT, RECORDING_WIDTH))
        mock_screen2 = np.array(np.random.randint(0,255, RECORDING_WIDTH * RECORDING_HEIGHT), dtype="uint8")
        mock_screen2 = mock_screen2.reshape((RECORDING_HEIGHT, RECORDING_WIDTH))

        reward1 = 1

        solver = RNNAgent(RECORDING_HEIGHT*RECORDING_WIDTH, len(list_inputs))

        action1 = solver.get_action(mock_screen1)

        solver.remember(mock_screen1, action1, reward1, mock_screen2)

        solver.batch_size = 1

        solver.experience_replay()

if __name__ == '__main__':
    unittest.main()
