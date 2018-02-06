import os
import numpy as np
import cv2

from PIL import Image
from color_utils import get_palette


def pil_to_cv2(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return img / 255

def display_palette(palette):
    num_colors = len(palette)
    colors = np.zeros([32, 32*num_colors, 3], dtype=np.float32)
    for color in range(num_colors):
        colors[0:32, 32*color:32*(color+1)] = palette[color]
    return colors

def get_B_img(img):
    w, h = img.size
    return img.crop((w/2, 0, w, h))

if __name__ == '__main__':
    base_path = 'datasets/edges2shoes/train/'

    for path in os.listdir(base_path):
        img = get_B_img(Image.open(base_path + path))
        palette = get_palette(img, color_count=6)
        cv2.imshow(path, pil_to_cv2(img))
        cv2.imshow('{} palette'.format(path), pil_to_cv2(display_palette(palette)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
