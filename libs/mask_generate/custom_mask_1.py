
import argparse
import numpy as np
import random
from PIL import Image

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]

class CustomMaskGenerator():
    def __init__(self, height=512, width=512, num_lines=20, num_circles=20, num_elips=20, rand_seed=None, channels=3):
        self.height = height
        self.width = width
        self.channels = channels
        self.num_lines = num_lines
        self.num_circles = num_circles
        self.num_elips = num_elips

        if rand_seed:
            seed(rand_seed)

    def random_walk(canvas, ini_x, ini_y, length):
        x = ini_x
        y = ini_y
        img_size = canvas.shape[-1]
        x_list = []
        y_list = []
        for i in range(length):
            r = random.randint(0, len(action_list) - 1)
            x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
            y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
            x_list.append(x)
            y_list.append(y)
        canvas[np.array(x_list), np.array(y_list)] = 0
        return canvas
