import unittest

from environment import Environment

class Test_environment(unittest.TestCase):
    def test_creation(self):
        env = Environment("Smash Melee")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
