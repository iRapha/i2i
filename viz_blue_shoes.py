import os
import cv2
import numpy as np
import colorsys

from shutil import copyfile
from PIL import Image


def get_only_blue():
    im = np.array(Image.open('summaries/sorted/sorted_shoes_train_9.png'))

    # get only the blue px
    im = im[1470:1545:15, :]
    im[0,0:229] = [0, 0, 0]
    im[-1,-492:] = [0, 0, 0]

    onlyblue = []
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if all(im[row, col] == [0, 0, 0]):
                continue
            onlyblue.append(list(im[row, col]))
    return onlyblue # shape should be (9929, 3)

def sort_by_var(onlyblue, saveimg=False):
    # SORT BY VARIANCE
    onlyblue.sort(key=np.var)

    if not saveimg:
        return onlyblue

    # VISUALIZE
    num_ex = len(onlyblue)

    px_height = 15
    height = int(np.ceil(np.sqrt(num_ex/px_height)))
    width = height * px_height

    sprite = np.zeros((height * px_height, width, 3), np.uint8)

    for i, pixels in enumerate(onlyblue):
        x = i % width
        y = (i // width) * px_height

        sprite[y:y+px_height, x, :] = pixels

    im = Image.fromarray(sprite)
    im.save('summaries/sorted/sorted_{}_byvariance.png'.format('onlyblue'))

def cp_into_folder(onlyblue):
    # cp all blue shoes into new folder
    with open('summaries/color/shoes_train_files.csv', 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if line is '': continue
            line_split = line.split(', ')
            px = list(int(x) for x in line_split[0:3])
            weight = float(line_split[3])
            filename = line_split[4]
            if px in onlyblue:
                copyfile('datasets/edges2shoes/train/{}'.format(filename),
                         'summaries/onlyblueshoes/{}_{}_{}'.format(np.var(px), weight, filename))

def get_blue_filenames(onlyblue, threshhold=0.0):
    filenames = {}
    with open('summaries/color/shoes_train_files.csv', 'r') as f:
        for line in f.read().split('\n'):
            if line is '': continue
            line_split = line.split(', ')

            px = list(int(x) for x in line_split[0:3])
            weight = float(line_split[3])
            filename = line_split[4]

            if px in onlyblue:
                if filename not in filenames:
                    filenames[filename] = 0.0
                filenames[filename] += weight

    return [filename for filename, weight in filenames if weight >= threshhold]

def get_blue_filenames_fromfolder(threshhold=0.0):
    filenames = {}
    for filename_ in os.listdir('summaries/onlyblueshoes'):
        _, weight, name, ext = filename_.split('_')
        weight = float(weight)
        filename = name + '_' + ext

        if filename not in filenames:
            filenames[filename] = 0.0
        filenames[filename] += weight

    return [filename for filename, weight in filenames.items() if weight >= threshhold]

if __name__ == '__main__':
    onlyblue = get_only_blue()
    onlyblue = sort_by_var(onlyblue)[int(0.6*len(onlyblue)):]
    # cp_into_folder(onlyblue)
    # filenames = get_blue_filenames(onlyblue, threshhold=0.05)
    filenames = get_blue_filenames_fromfolder(threshhold=0.05)
    with open('summaries/color/blueshoes.csv', 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')
