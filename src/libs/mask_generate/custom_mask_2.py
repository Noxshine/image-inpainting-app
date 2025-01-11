import random

import numpy as np

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]

class CustomMaskGenerator2():
    def __init__(self, size_x=512, size_y=512):
        self.size_x = size_x
        self.size_y = size_y
        self.canvas = np.ones((size_x, size_y)).astype("i")
        self.ini_x = random.randint(0, size_x - 1)
        self.ini_y = random.randint(0, size_y - 1)
        self.length=size_x**2

    def _random_walk(self):
        x = self.ini_x
        y = self.ini_y
        img_size = self.canvas.shape[-1]
        x_list = []
        y_list = []
        for i in range(self.length):
            r = random.randint(0, len(action_list) - 1)
            x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
            y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
            x_list.append(x)
            y_list.append(y)
        self.canvas[np.array(x_list), np.array(y_list)] = 0
        return self.canvas
