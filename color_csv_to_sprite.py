"""Converts a csv of pixels into a sprite image and a tsv for projector."""
import numpy as np

from PIL import Image


dataset = 'handbags_val'

csv_filename = 'summaries/color/{}.csv'.format(dataset)
tsv_filename = 'summaries/color/{}.tsv'.format(dataset)
sprite_filename = 'summaries/color/{}.png'.format(dataset)

with open(csv_filename, 'r') as f:
    all_colors = f.read().split('\n')
    color_count = len(all_colors)

sprite_len = int(np.ceil(np.sqrt(color_count))) # sprite image width and height
example_len = int(5) # each sprite is 5x5 px

sprite = np.zeros((sprite_len * example_len, sprite_len * example_len, 3), np.uint8)
tsv = "pxid\trgb\tweight\n"

for i, line in enumerate(all_colors):
    if line is '': continue
    if i % 10 == 0: print('{}/{}'.format(i, color_count), end='\r')

    line_split = line.split(', ')
    pixels = [int(x) for x in line_split[0:3]]
    weight = float(line_split[3])

    tsv += "{}\t{}\t{}\n".format(i, repr(pixels), weight)

    x = (i * example_len) % (sprite_len * example_len)
    y = ((i * example_len) // (sprite_len * example_len)) * example_len

    sprite[y:y+example_len, x:x+example_len, :] = pixels

print('') # new line

im = Image.fromarray(sprite)
im.save(sprite_filename)

with open(tsv_filename, 'w') as f:
    f.write(tsv)
