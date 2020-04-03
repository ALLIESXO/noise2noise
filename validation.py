# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import numpy as np
import PIL.Image
import tensorflow as tf
import imageio, cv2
imageio.plugins.freeimage.download()

import dnnlib
import dnnlib.submission.submit as submit
import dnnlib.tflib.tfutil as tfutil
from dnnlib.tflib.autosummary import autosummary

import util
import config

def filter_clean_with_features(path) -> bool:
    if "clean" in path:
        if "albedo" in path:
            return False
        elif "normal" in path:
            return False
        else: 
            return True

    return True

def load_image(fname):
    if ".exr" in fname:
        im = imageio.imread(fname)
        im = im[:,:,:3] # remove alpha channel
        arr = np.array(im, dtype=np.float32)
        assert len(arr.shape) == 3
        return arr.transpose([2, 0, 1]) # hwc to chw  
    else:
        im = PIL.Image.open(fname)
        arr = np.array(im.convert('RGB'), dtype=np.float32)
        assert len(arr.shape) == 3
        return arr.transpose([2, 0, 1]) / 255.0

class ValidationSet:
    def __init__(self, submit_config):
        self.images = None
        self.submit_config = submit_config
        return

    def load(self, dataset_dir):
        import glob
        exrVal = False
        fnames = sorted(glob.iglob(os.path.join(dataset_dir) + '**/*.png', recursive=True))
        fnames = fnames + sorted(glob.iglob(os.path.join(dataset_dir) + '**/*.exr', recursive=True))
        fnames = list(filter(filter_clean_with_features, fnames))
        for fname in fnames: 
            print(fname)
            print(exrVal)
            if ".exr" in fname:
                exrVal = True
                
        # now the first path is always the clean one and the last three paths are the noisy with feature paths
        if len(fnames) == 0:
            print ('\nERROR: No files found using the following glob pattern:', os.path.join(dataset_dir), '\n')
            sys.exit(1)

        images = []
        for clean, noisy_img, albedo_img, normal_img in zip(fnames[0::4], fnames[1::4], fnames[2::4], fnames[3::4]):
            try:
                clean = load_image(clean)
                noisy = load_image(noisy_img)
                albedo = load_image(albedo_img)
                normal = load_image(normal_img)
                noisy = np.append(noisy,albedo,axis=0) 
                noisy = np.append(noisy,normal,axis=0)
                reshaped = (clean, noisy, exrVal)
                images.append(reshaped) 
            except OSError as e:
                print ('Skipping file due to error: ', e)
        self.images = images


    def evaluate(self, net, iteration, noise_func):
        avg_psnr = 0.0
        # self.images should be a tuple of images -> snd::noisy image with 9th dimension ; fst::clean
        for idx in range(len(self.images)):
            img_pair = self.images[idx]
            exrVal = img_pair[2]
            noisy_img = img_pair[1]
            orig_img = img_pair[0]

            w = orig_img.shape[2]
            h = orig_img.shape[1]

            orig255 = orig_img
            #exr image prediction
            pred255 = util.infer_image(net,noisy_img)
            if exrVal is False:
                #pred255 = util.infer_image(net, noisy_img)
                pred255 = util.clip_to_uint8(pred255)
                orig255 = util.clip_to_uint8(orig_img)
                noisy_img = util.clip_to_uint8(noisy_img[0:3,:,:])

            # the best would be if the input x is already tonemapped 
            assert (pred255.shape[2] == w and pred255.shape[1] == h)
            
            if exrVal is False:
                sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
                s = np.sum(sqerr)
                cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
                avg_psnr += cur_psnr
            else:
                cur_psnr = cv2.PSNR(pred255, orig_img) # TODO: try this variant else just map them to png images 
                avg_psnr += cur_psnr

            #meansq_error = np.mean(np.square(orig255.astype(np.float32) - pred255.astype(np.float32))/np.square(pred255.astype(np.float32)+0.01))

            util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))
        #print ('Validation Loss: %.2f' % autosummary('Validation_Loss', meansq_error))
 

def validate(submit_config: dnnlib.SubmitConfig, noise: dict, dataset: dict, network_snapshot: str):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**dataset)

    ctx = dnnlib.RunContext(submit_config, config)

    tfutil.init_tf(config.tf_config)

    with tf.device("/gpu:0"):
        net = util.load_snapshot(network_snapshot)
        validation_set.evaluate(net, 0, noise_augmenter.add_validation_noise_np)
    ctx.close()

#TODO: add infer image for monte carlo options 
def infer_image(network_snapshot: str, image: str, out_image: str):
    tfutil.init_tf(config.tf_config)
    net = util.load_snapshot(network_snapshot)
    im = PIL.Image.open(image).convert('RGB')
    arr = np.array(im, dtype=np.float32)
    reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
    pred255 = util.infer_image(net, reshaped)
    t = pred255.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(out_image))
    print ('Inferred image saved in', out_image)
