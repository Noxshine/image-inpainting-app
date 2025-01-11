import argparse
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np

from libs.MAT.datasets.mask_generator_256 import RandomMask
from libs.mask_generate.custom_mask_1 import CustomMaskGenerator1
from libs.mask_generate.custom_mask_2 import CustomMaskGenerator2


def mask_generate(type:int, img, mask_dir, masked_dir):
    mask=None
    if type==1:
        # Instantiate mask generator
        mask_generator = CustomMaskGenerator1(512, 512, rand_seed=None, channels=3)

        # Load mask
        mask = mask_generator._generate_mask()

    elif type==2:
        # Instantiate mask generator
        mask_generator = CustomMaskGenerator2()

        # Load mask
        mask = mask_generator._random_walk()

    elif type==3:
        mask = RandomMask(512)
        mask = (mask[0] * 255).astype(np.uint8)  # Extract the first channel and scale


    masked_img = deepcopy(img)
    masked_img[mask==0] = 255

    # save mask
    cv2.imwrite(mask_dir, mask * 255)

    # save masked image
    cv2.imwrite(masked_dir, masked_img)

  # Show side by side
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(img)
    axes[1].imshow(mask*255)
    axes[2].imshow(masked_img)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=1)
    parser.add_argument('--image_dir', type=str, default='../../data/image-test.jpg')
    parser.add_argument('--mask_dir', type=str, default='../../data/mask.png')
    parser.add_argument('--masked_dir', type=str, default='../../data/masked-img.png')
    args = parser.parse_args()

    img = cv2.imread(args.image_dir)
    shape = img.shape

    # resize to 512x512
    if shape[0] != 512 or shape[1] != 512:
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(args.image_dir, img)

    # generate mask and image with mask to ./data
    mask_generate(args.type, img, args.mask_dir, args.masked_dir)
