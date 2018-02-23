import os
import numpy as np
import cv2

from PIL import Image
from color_utils import get_palette
from scipy.spatial.distance import cdist
from img_utils import pil_to_cv2, get_B_img


def display_palette(palette, percentages=None):
    """Returns an image representation of the palette for display."""
    num_colors = len(palette)
    colors = np.zeros([32, 32*num_colors, 3], dtype=np.float32)
    for color in range(num_colors):
        colors[0:32, 32*color:32*(color+1)] = palette[color]
        if percentages is not None:
            txt = str(percentages[color])
            txt_color = list(map(lambda x: 255 - x, palette[color]))
            cv2.putText(colors, txt, (2+32*color,16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color)
    return colors

def palettify(img, palette):
    """Returns an image w/ each pixel as its nearest neighbor in the palette."""
    img = np.array(img)
    # we add white because we want to ignore it
    palette = np.array(palette + [[255, 255, 255]])

    closest_points = cdist(img.reshape((-1, 3)), palette)
    palette_idx = np.argmin(closest_points, axis=1).astype(np.int8)
    palettified = palette[palette_idx]
    return palettified.reshape(img.shape).astype(np.uint8)

def get_palette_percentages(img, palette):
    """Returns the % of pixels in img that are NNs of each color in palette."""
    img = np.array(img)
    # we add white because we want to ignore it
    palette = np.array(palette + [[255, 255, 255]])

    closest_points = cdist(img.reshape((-1, 3)), palette)
    palette_idx = np.argmin(closest_points, axis=1).astype(np.int8)
    counts = np.unique(palette_idx, return_counts=True)[1][:-1] # remove white
    return counts / sum(counts)

def visualize_all(base_path):
    """For each image, computes the palette and displays palettified image."""
    for path in os.listdir(base_path):
        img = get_B_img(Image.open(base_path + path))
        palette = get_palette(img, color_count=6)
        palette_percentages = get_palette_percentages(img, palette)

        visualize_one(path, img, palette, palette_percentages)

def visualize_one(path, img, palette, palette_percentages, save=False):
    og_img = pil_to_cv2(img)
    palette_img = pil_to_cv2(display_palette(palette, percentages=palette_percentages))
    palletified_img = pil_to_cv2(palettify(img, palette))
    if save:
        cv2.imwrite(path + '.png', og_img)
        cv2.imwrite('{}_palette.png'.format(path), palette_img)
        cv2.imwrite('{}_palettified.png'.format(path), palletified_img)
        return
    cv2.namedWindow(path)
    cv2.moveWindow(path, 30, 175)
    cv2.namedWindow('{} palette'.format(path))
    cv2.moveWindow('{} palette'.format(path), 30, 30)
    cv2.namedWindow('{} palettified'.format(path))
    cv2.moveWindow('{} palettified'.format(path), 30, 500)

    cv2.imshow(path, og_img)
    cv2.imshow('{} palette'.format(path), palette_img)
    cv2.imshow('{} palettified'.format(path), palletified_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    base_path = 'datasets/edges2handbags/train/'
    all_colors_file = 'summaries/color/handbags_train.csv'

    num_imgs = len(os.listdir(base_path))

    with open(all_colors_file, 'w') as f:
        for i, path in enumerate(os.listdir(base_path)):
            if i % 10 == 0:
                print('{}/{}'.format(i, num_imgs), end='\r')

            img = get_B_img(Image.open(base_path + path))
            palette = get_palette(img, color_count=6)
            palette_percentages = get_palette_percentages(img, palette)
            for color, weight in zip(palette, palette_percentages):
                f.write('{}, {}, {}, {}\n'.format(*color, weight))
        print('') # new line
