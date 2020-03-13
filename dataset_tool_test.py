import PIL.Image
import tensorflow as tf
import numpy as np
from collections import defaultdict

size_stats = defaultdict(int)
format_stats = defaultdict(int)

im = PIL.Image.open("D:\\Bachelorarbeit\\noise2noise\\results\\00012-autoencoder-n2n\\img_0_x.png")
format_stats[im.mode] += 1
if (im.width < 256 or im.height < 256):
    size_stats['< 256x256'] += 1
else:
    size_stats['>= 256x256'] += 1
arr = np.array(im.convert('RGB'), dtype=np.uint8)

pix = im.load()
print(str(pix[25,25][0]/255))
assert len(arr.shape) == 3
print(arr.shape)
arr = arr.transpose([2, 0, 1])
print(arr.shape)

