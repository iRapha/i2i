import os

from PIL import Image
from img_utils import get_B_img


# this will copy all B shoes (except light-blue shoes) and all B handbags
# into trainA and trainB respectively
# it will also copy light-blue shoes into testA
# and light-blue handbags into testB

if __name__ == '__main__':
    base_dir = 'datasets/noblueshoes/'
    shoes_train = 'datasets/edges2shoes/train/'
    handbags_train = 'datasets/edges2handbags/train/'
    blueshoes_file = 'summaries/color/blueshoes.csv'
    bluehandbags_file = 'summaries/color/bluehandbags.csv'

    blueshoes = set(open(blueshoes_file, 'r').read().split('\n'))
    bluehandbags = set(open(bluehandbags_file, 'r').read().split('\n'))

    # FIRST: SHOES
    num_imgs = len(os.listdir(shoes_train))
    for i, path in enumerate(os.listdir(shoes_train)):
        if i % 10 == 0:
            print('{}/{}'.format(i, num_imgs), end='\r')

        # read and crop image
        img = get_B_img(Image.open(shoes_train + path))

        # determine where to save it:
        dest = base_dir + 'trainA/' + path
        if path in blueshoes:
            dest = base_dir + 'testA/' + path

        # save image
        img.save(dest)
    print('') # new line

    # SECOND: HANDBAGS
    num_imgs = len(os.listdir(handbags_train))
    for i, path in enumerate(os.listdir(handbags_train)):
        if i % 10 == 0:
            print('{}/{}'.format(i, num_imgs), end='\r')

        # read and crop image
        img = get_B_img(Image.open(handbags_train + path))

        # determine where to save it:
        dest = base_dir + 'trainB/' + path

        # TODO: it'll be interesting to see if no blue handbags exist either,
        # what will happen to test case?
        #  if path in blueshoes:
            #  dest = base_dir + 'testB/' + path

        # save image
        img.save(dest)
    print('') # new line
