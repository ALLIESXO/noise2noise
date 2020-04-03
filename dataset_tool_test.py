import PIL.Image
import tensorflow as tf
import numpy as np
from collections import defaultdict
import numpy 
import math
import cv2


def main():
    original = cv2.imread("D:\\Bachelorarbeit\\noise2noise\\datasets\\monte_carlo\\0_2.3camp\\clean\\Image0001.png")
    contrast = cv2.imread("D:\\Bachelorarbeit\\noise2noise\\datasets\\monte_carlo\\0_2.3camp\\noisy\\Image0001.png")
    #print(psnr(original,contrast))
    psnr_val = cv2.PSNR(original, contrast)
    print(psnr_val)

def clip_to_uint8(arr):
    return np.clip((arr ) * 255.0 , 0, 255).astype(np.uint8)

def load_image(fname):
    im = PIL.Image.open(fname)
    arr = np.array(im.convert('RGB'), dtype=np.float32)
    assert len(arr.shape) == 3
    return arr.transpose([2, 0, 1]) / 255.0

def testShapes():
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


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def testPSNR(): 
    avg_psnr = 0
    image1 = load_image("D:\\Bachelorarbeit\\noise2noise\\datasets\\monte_carlo\\0_2.3camp\\clean\\Image0001.png")
    image2 = load_image("D:\\Bachelorarbeit\\noise2noise\\datasets\\monte_carlo\\0_2.3camp\\noisy\\Image0001.png")
    
    w = image1.shape[2]
    h = image1.shape[1]

    image1 = clip_to_uint8(image1)
    image2 = clip_to_uint8(image2)

    print("w: " + str(w))
    
    print("h: " + str(h))

    sqerr = np.square(image2.astype(np.float32) - image1.astype(np.float32))
    s = np.sum(sqerr)
    print("s: " + str(s))
    cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
    avg_psnr += cur_psnr

    print ("PSNR: " + str(avg_psnr))

if __name__ == "__main__":
    main()
