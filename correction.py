import numpy as np
from PIL import Image


def conv(img):
    img_arr = np.array(img)
    bright_img = img_arr * 1.2
    contrast_img = np.clip((bright_img - 127) * 1.5 +
                           127, 0, 255).astype(np.uint8)
    return contrast_img
