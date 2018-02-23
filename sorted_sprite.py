import colorsys
import numpy as np

from PIL import Image


# LOAD ALL PALETTE PIXELS
dataset = 'handbags_train'
with open('summaries/color/{}.csv'.format(dataset), 'r') as f:
    csv = f.read().split('\n')
    N = len(csv) - 1 # Number of items.
    D = 3 # Dimensionality of the embedding.
    all_ex = []
    for i, ex in enumerate(csv):
        if ex == '': continue
        ex_split = ex.split(', ')
        all_ex.append([int(x) for x in ex_split[0:3]])

# STEP SORTING
def step (r,g,b, repetitions=1):
    lum = np.sqrt(.241 * r + .691 * g + .068 * b)

    h, s, v = colorsys.rgb_to_hsv(r,g,b)

    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)

    return (h2, lum, v2)
all_ex.sort(key=lambda rgb: step(rgb[0], rgb[1], rgb[2], 8))

# VISUALIZE
num_ex = len(all_ex)

px_height = 15
height = int(np.ceil(np.sqrt(num_ex/px_height)))
width = height * px_height

sprite = np.zeros((height * px_height, width, 3), np.uint8)

for i, pixels in enumerate(all_ex):
    x = i % width
    y = (i // width) * px_height

    sprite[y:y+px_height, x, :] = pixels

im = Image.fromarray(sprite)
im.save('summaries/sorted/sorted_{}.png'.format(dataset))
