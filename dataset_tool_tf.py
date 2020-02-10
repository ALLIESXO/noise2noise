# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import glob
import os
import sys
import argparse
import tensorflow as tf

import PIL.Image
import numpy as np

from collections import defaultdict

size_stats = defaultdict(int)
format_stats = defaultdict(int)

def load_image(fname):
    global format_stats, size_stats
    im = PIL.Image.open(fname)
    format_stats[im.mode] += 1
    if (im.width < 256 or im.height < 256):
        size_stats['< 256x256'] += 1
    else:
        size_stats['>= 256x256'] += 1
    arr = np.array(im.convert('RGB'), dtype=np.uint8)
    assert len(arr.shape) == 3
    return arr.transpose([2, 0, 1])

def shape_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

examples='''examples:

  python %(prog)s --input-dir=./kodak --out=imagenet_val_raw.tfrecords
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a TensorFlow tfrecords training set.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", help="Directory containing ImageNet images")
    parser.add_argument("--out", help="Filename of the output tfrecords file")
    parser.add_argument("--nofeatures", help="Ignores Albedo and Normal Image in the given subdirectories")
    args = parser.parse_args()

    if args.input_dir is None:
        print ('Must specify input file directory with --input-dir')
        sys.exit(1)
    if args.out is None:
        print ('Must specify output filename with --out')
        sys.exit(1)
    if args.nofeatures is not None:
        print ('Ignoring Features')

    print ('Loading image list from %s' % args.input_dir)
    images = sorted(glob.iglob(os.path.join(args.input_dir) + '**/*.JPEG', recursive=True))
    images += sorted(glob.iglob(os.path.join(args.input_dir) + '**/*.png', recursive=True))
    images += sorted(glob.iglob(os.path.join(args.input_dir) + '**/*.jpg', recursive=True))

    if args.nofeatures is not None: # ignore all paths which contain albedo or normal as information 
        for image in images:
            if "normal" in image:
                images.remove(image)
            if "albedo" in image:
                images.remove(image)
            print("Removed " + str(image) + " from tfRecord. (nofeatures)")

    #----------------------------------------------------------
    outdir = os.path.dirname(args.out)
    os.makedirs(outdir, exist_ok=True)
    writer = tf.io.TFRecordWriter(args.out)

    if args.nofeatures is not None: # ignore the featurees 
        for imgname, imgname2 in zip(images[0::2], images[1::2]):
            print(str(imgname))
            print(str(imgname2))
            image = load_image(imgname)
            image2 = load_image(imgname2)
            feature = {
            'shape': shape_feature(image.shape),
            'data': bytes_feature(tf.compat.as_bytes(image.tostring())),
            'shape2': shape_feature(image2.shape),
            'data2': bytes_feature(tf.compat.as_bytes(image2.tostring()))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    
    else: 
        for noisy_img1, noisy_img2, albedo_img, normal_img in zip(images[0::4], images[1::4], images[2::4], images[3::4]):
            print(str(noisy_img1))
            print(str(noisy_img2))
            print(str(albedo_img))
            print(str(normal_img))
            print("#######################################")
            noisy_img1 = load_image(noisy_img1)
            noisy_img2 = load_image(noisy_img2)
            albedoFeature = load_image(albedo_img)
            normalFeature = load_image(normal_img)

            noisy_img1 = np.append(noisy_img1,albedoFeature,axis=0)
            noisy_img1 = np.append(noisy_img1,normalFeature,axis=0)
            print(noisy_img1.shape)
            noisy_img2 = np.append(noisy_img2,albedoFeature,axis=0)
            noisy_img2 = np.append(noisy_img2,normalFeature,axis=0)

            feature = {
            'shape': shape_feature(noisy_img1.shape),
            'data': bytes_feature(tf.compat.as_bytes(noisy_img1.tostring())),
            'shape2': shape_feature(noisy_img2.shape),
            'data2': bytes_feature(tf.compat.as_bytes(noisy_img2.tostring())),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print ('Dataset statistics:')
    print ('  Formats:')
    for key in format_stats:
        print ('    %s: %d images' % (key, format_stats[key]))
    print ('  width,height buckets:')
    for key in size_stats:
        print ('    %s: %d images' % (key, size_stats[key]))
    writer.close()



if __name__ == "__main__":
    main()
