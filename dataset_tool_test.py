import PIL.Image
import tensorflow as tf
import numpy as np
from collections import defaultdict

size_stats = defaultdict(int)
format_stats = defaultdict(int)

im = PIL.Image.open("2spp-1000houses/house1/0_0.0camp/Image0001.png")
format_stats[im.mode] += 1
if (im.width < 256 or im.height < 256):
    size_stats['< 256x256'] += 1
else:
    size_stats['>= 256x256'] += 1
arr = np.array(im.convert('RGB'), dtype=np.uint8)
assert len(arr.shape) == 3
print(arr.shape)
arr = arr.transpose([2, 0, 1])
print(arr.shape)
