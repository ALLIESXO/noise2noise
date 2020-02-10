# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
def parse_tfrecord_tf_with_features(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string),
        'shape2': tf.FixedLenFeature([3], tf.int64),
        'data2': tf.FixedLenFeature([], tf.string)})

    image1 = tf.decode_raw(features['data'], tf.uint8)
    image2 = tf.decode_raw(features['data2'], tf.uint8)

    return (tf.reshape(image1, features['shape']),tf.reshape(image2, features['shape2']))

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string),
        'shape2': tf.FixedLenFeature([3], tf.int64),
        'data2': tf.FixedLenFeature([], tf.string),})

    data = tf.decode_raw(features['data'], tf.uint8)
    data2 = tf.decode_raw(features['data2'], tf.uint8)
    return (tf.reshape(data, features['shape']),tf.reshape(data2, features['shape2']))

# [c,h,w] -> [h,w,c]
def chw_to_hwc(x):
    return tf.transpose(x, perm=[1, 2, 0])

# [h,w,c] -> [c,h,w]
def hwc_to_chw(x):
    return tf.transpose(x, perm=[2, 0, 1])

def resize_small_image(x):
    shape = tf.shape(x)
    return tf.cond(
        tf.logical_or(
            tf.less(shape[2], 256),
            tf.less(shape[1], 256)
        ),
        true_fn=lambda: hwc_to_chw(tf.image.resize_images(chw_to_hwc(x), size=[256,256], method=tf.image.ResizeMethod.BICUBIC)),
        false_fn=lambda: tf.cast(x, tf.float32)
     )

def random_crop_noised_clean(x, add_noise):
    cropped = tf.random_crop(resize_small_image(x), size=[3, 256, 256]) / 255.0 - 0.5
    return (add_noise(cropped), add_noise(cropped), cropped)

def random_crop_monte_carlo(x,y, useFeatures):
    if useFeatures is True:
        cropped_noisy_input = tf.random_crop(resize_small_image(x), size=[9, 256, 256]) / 255.0 - 0.5
        cropped_noisy_target = tf.random_crop(resize_small_image(y), size=[9, 256, 256]) / 255.0 - 0.5
    else: 
        cropped_noisy_input = tf.random_crop(resize_small_image(x), size=[3, 256, 256]) / 255.0 - 0.5
        cropped_noisy_target = tf.random_crop(resize_small_image(y), size=[3, 256, 256]) / 255.0 - 0.5

    return (cropped_noisy_input,cropped_noisy_target)

def create_dataset(train_tfrecords, minibatch_size, add_noise):
    print ('Setting up dataset source from', train_tfrecords)
    buffer_mb   = 256
    num_threads = 2
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb<<20)
    dset = dset.repeat()  # .repeat(count=None), repeats this dataset indefinitely amount of times 
    buf_size = 1000 
    dset = dset.prefetch(buf_size)
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
    dset = dset.shuffle(buffer_size=buf_size)
    dset = dset.map(lambda x: random_crop_noised_clean(x, add_noise))
    dset = dset.batch(minibatch_size)
    it = dset.make_one_shot_iterator()
    return it

def create_monte_carlo_dataset(train_tfrecords, minibatch_size, add_noise, useFeatures):
    print ('Setting up dataset source from', train_tfrecords)
    buffer_mb   = 256
    num_threads = 2
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb<<20)
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/TFRecordDataset
    dataset_it_length = sum(1 for _ in tf.python_io.tf_record_iterator(train_tfrecords))
    print("!!!!!!!!!!!!!!!!!!!!!The number of iterations per epoch is: " + str(dataset_it_length))
    dset = dset.repeat()
    buf_size = 1000
    dset = dset.prefetch(buf_size) # not sure if I need to comment it out or not.
    if useFeatures is True:
        dset = dset.map(parse_tfrecord_tf_with_features, num_parallel_calls=num_threads)
    else:
        dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
    #dset = dset.shuffle(buffer_size=buf_size) 
    dset = dset.map(lambda x,y: random_crop_monte_carlo(x,y, useFeatures))
    dset = dset.batch(minibatch_size)
    it = dset.make_one_shot_iterator()
    return it