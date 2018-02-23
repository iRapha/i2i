import cv2
import numpy as np

def pil_to_cv2(img):
    """Converts a numpy image into a cv2-display-suitable representation."""
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return img / 255

def get_B_img(img):
    """Returns the right half of an image."""
    w, h = img.size
    return img.crop((w/2, 0, w, h))

