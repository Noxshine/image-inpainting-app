
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt

from libs.mask_generate.custom_mask_1 import CustomMaskGenerator1
from libs.mask_generate.custom_mask_2 import CustomMaskGenerator2


def mask_generate(type:int, img):
    mask=None
    if type==1:
        # Instantiate mask generator
        mask_generator = CustomMaskGenerator1(512, 512, rand_seed=None, channels=3)

        # Load mask
        mask = mask_generator._generate_mask()

        # Image + mask
        masked_img = deepcopy(img)
        masked_img[mask==0] = 255

        cv2.imwrite('../../data/mask_image.png', masked_img)
        cv2.imwrite('../../data/mask.png', mask)

        # Show side by side
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(img)
        axes[1].imshow(mask*255)
        axes[2].imshow(masked_img)
        plt.show()

    elif type==2:
        # Instantiate mask generator
        mask_generator = CustomMaskGenerator2()

        # Load mask
        mask = mask_generator._random_walk()

        cv2.imwrite('../../data/mask.png', mask)

        # Show side by side
        # _, axes = plt.subplots(1, 1, figsize=(5, 5))
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # axes.imshow(mask)
        # plt.show()

    masked_img = deepcopy(img)
    masked_img[mask==0] = 255
    cv2.imwrite('../../data/masked_img.png', masked_img)
    cv2.imwrite('../../data/mask.png', mask)

if __name__ == "__main__":
    img = cv2.imread('../../data/image_test.png')
    shape = img.shape

    # resize to 512x512
    if shape[0] != 512 or shape[1] != 512:
        img = cv2.resize(img, (512, 512))
        cv2.imwrite('../../data/image_test.png', img)

    # generate mask and image with mask to ./data
    mask_generate(2, img)
