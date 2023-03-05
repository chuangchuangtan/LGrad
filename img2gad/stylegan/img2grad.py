# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '1'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
random_seed = 1000


def main():
    # Initialize TensorFlow.
    tflib.init_tf()
    args = sys.argv
    num_gpus       = int(args[1]) if len(args)>1 else 1
    input_dir      = args[2]      if len(args)>2 else './genimg'
    out_dir        = args[3]      if len(args)>3 else './genimg_grad'
    pkl_path       = args[4]      if len(args)>4 else 'networks/karras2019stylegan-bedrooms-256x256.pkl'
    minibatch_size = int(args[5]) if len(args)>5 else 1
    os.makedirs(out_dir, mode=0o777, exist_ok=True)
    nowpath = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(nowpath, pkl_path), 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    _D.run_img2grad(input_dir, out_dir, minibatch_size=minibatch_size, num_gpus = num_gpus)


if __name__ == "__main__":
    print()
    print('='*15)
    print('  '.join(list(sys.argv)))
    main()
