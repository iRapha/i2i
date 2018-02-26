import os
import cv2
import numpy as np
import colorsys

from shutil import copyfile
from PIL import Image
from img_utils import pil_to_cv2


dataset = 'handbags' # or 'shoes'


def get_only_blue():
    if dataset == 'shoes':
        im = np.array(Image.open('summaries/sorted/sorted_shoes_train_9.png'))
        # get only the blue px
        im = im[1470:1545:15, :]
        im[0,0:229] = [0, 0, 0]
        im[-1,-492:] = [0, 0, 0]
    else:
        im = np.array(Image.open('summaries/sorted/sorted_handbags_train_9.png'))
        # get only the blue px
        im = im[2145:2385:15, :]
        im[0,0:1019] = [0, 0, 0]
        im[-1,-2321:] = [0, 0, 0]

    onlyblue = []
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if all(im[row, col] == [0, 0, 0]):
                continue
            onlyblue.append(list(im[row, col]))
    return onlyblue # shape should be (9929, 3) (if shoes)

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
    if dataset == 'shoes':
        im.save('summaries/sorted/sorted_{}_byvariance.png'.format('onlyblueshoes'))
    else:
        im.save('summaries/sorted/sorted_{}_byvariance.png'.format('onlybluehandbags'))

def cp_into_folder(onlyblue):
    # cp all blue shoes into new folder
    if dataset == 'shoes':
        csv_filename = 'summaries/color/shoes_train_files.csv'
    else:
        csv_filename = 'summaries/color/handbags_train_files.csv'

    onlyblue_set = set(tuple(x) for x in onlyblue)

    with open(csv_filename, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if line is '': continue
            line_split = line.split(', ')
            px = tuple(int(x) for x in line_split[0:3])
            weight = float(line_split[3])
            filename = line_split[4]
            if px in onlyblue_set:
                copyfile('datasets/edges2{}/train/{}'.format(dataset, filename),
                         'summaries/onlyblue{}/{}_{}_{}'.format(handbags, np.var(px), weight, filename))

def get_blue_filenames(onlyblue, threshhold=0.0):
    filenames = {}
    if dataset == 'shoes':
        csv_filename = 'summaries/color/shoes_train_files.csv'
    else:
        csv_filename = 'summaries/color/handbags_train_files.csv'

    with open(csv_filename, 'r') as f:
        num_lines = len(f.read().split('\n'))

    onlyblue_set = set(tuple(x) for x in onlyblue)

    with open(csv_filename, 'r') as f:
        for i, line in enumerate(f.read().split('\n')):
            if i % 10 == 0:
                print('{}/{}'.format(i, num_lines), end='\r')

            if line is '': continue

            line_split = line.split(', ')

            px = tuple(int(x) for x in line_split[0:3])
            weight = float(line_split[3])
            filename = line_split[4]

            if px in onlyblue_set:
                if filename not in filenames:
                    filenames[filename] = 0.0
                filenames[filename] += weight
        print('') # new line

    return [filename for filename, weight in filenames.items() if weight >= threshhold]

def get_blue_filenames_fromfolder(threshhold=0.0):
    # faster than get_blue_filenames but you must have run cp_into_folder first
    filenames = {}
    for filename_ in os.listdir('summaries/onlyblue{}'.format(dataset)):
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
    filenames = get_blue_filenames(onlyblue, threshhold=0.05)
    # filenames = get_blue_filenames_fromfolder(threshhold=0.05)
    with open('summaries/color/blue{}.csv'.format(dataset), 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')
