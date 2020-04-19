
# Noise2Noise: Learning Image Restoration without Clean Data - _Official TensorFlow implementation of the ICML 2018 paper_

**Jaakko Lehtinen**, **Jacob Munkberg**, **Jon Hasselgren**, **Samuli Laine**, **Tero Karras**, **Miika Aittala**, **Timo Aila**

**Abstract**:

_We apply basic statistical reasoning to signal reconstruction by machine learning -- learning to map corrupted observations to clean signals -- with a simple and powerful conclusion: it is possible to learn to restore images by only looking at corrupted examples, at performance at and sometimes exceeding training using clean data, without explicit image priors or likelihood models of the corruption. In practice, we show that a single model learns photographic noise removal, denoising synthetic Monte Carlo images, and reconstruction of undersampled MRI scans -- all corrupted by different processes -- based on noisy data only._

![alt text](img/n2nteaser_1024width.png "Denoising comparison")

## Resources

* [Paper (arXiv)](https://arxiv.org/abs/1803.04189)

All the material, including source code, is made freely available for non-commercial use under the Creative Commons [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license. Feel free to use any of the material in your own work, as long as you give us appropriate credit by mentioning the title and author list of our paper.

## Getting started

The below sections detail how to get set up for training the Noise2Noise network using the ImageNet validation dataset.   Noise2Noise MRI denoising instructions are at the end of this document.

### Python requirements

This code is tested with Python 3.6.  We're using [Anaconda 5.2](https://www.anaconda.com/download/) to manage the Python environment.  Here's how to create a clean environment and install library dependencies:

```
conda create -n n2n python=3.6
conda activate n2n
conda install tensorflow-gpu
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This will install TensorFlow and other dependencies used in this project.

### Preparing datasets for training and validation

This section explains how to prepare a dataset into a TFRecords file for use in training the Noise2Noise denoising network. The image denoising results presented in the Noise2Noise paper have been obtained using a network trained with the ImageNet validation set.

## Monte Carlo Denoising

Training on high dynamic range images is by default very unstable.
It is recommended to use batch normalization either before or after (will try both) each LeakyRelu. 
Also adjusting the learning rate using step decay after few epochs improved stability.
Following statistics show the training on non tonemapped inputs: 

<img src="https://raw.githubusercontent.com/ALLIESXO/noise2noise--monte-carlo-images/master/images/avr_psnr_x_y.PNG" width="441" height="301">

<img src="https://raw.githubusercontent.com/ALLIESXO/noise2noise--monte-carlo-images/master/images/Loss_x_y.PNG" width="473" height="308">

<img src="https://raw.githubusercontent.com/ALLIESXO/noise2noise--monte-carlo-images/master/images/training_rate.PNG" width="437" height="304">

Run with: 
python config.py train --train-tfrecords=datasets/yourpath.tfrecords --noise=monte_carlo --useFeatures=True --hdr=True
