import os
import cv2
import numpy as np
from random import randint, seed

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

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""
        img = np.zeros((self.height, self.width, self.channels), np.uint8)


        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 256 or self.height < 256:
            raise Exception("Width and Height of mask must be at least 256!")
        
        # Draw random lines
        for _ in range(randint(1, self.num_lines)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
        for _ in range(randint(1, self.num_circles)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
            
        # Draw random ellipses
        for _ in range(randint(1, self.num_elips)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
        
        return 1-img