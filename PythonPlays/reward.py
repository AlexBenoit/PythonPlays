from pubsub import pub

class reward(object):
    """description of class"""
    def __init__(self):
        self.value = 0

    def get_value(self):
        return self.value

    def update(self, value, position):
        self.value = value

if __name__ == "__main__":
    print("in main")