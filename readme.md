Image to Image Translation
==========================

Colorthief uses color quantization algorithm median-cut.

Here's how I think we're gonna be able to quantize the images.
1. get palette per image
2. for each non-white pixel in the image, replace it with its nn in the pallete
3. count the percentages of pixels of each color in the palette and save that
   along with the palette.

4. after all images have been processed, construct a new image that's 10x10
5. color each pixel according to the palette, maintaining percentages
6. stitch all images together and run get\_palette with num\_colors=large
then you'll have a palette of most common colors accross shoe images.
now do the same for handbags and find the colors that are most prevalent in one
but not the other.
