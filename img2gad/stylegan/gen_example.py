# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import sys
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
random_seed = 1000

def random_latents(num_latents, G, random_state=None):
    if random_state is not None:
        return random_state.randn(num_latents, *G.input_shape[1:]).astype(np.float32)
    else:
        return np.random.randn(num_latents, *G.input_shape[1:]).astype(np.float32)

def main():
    # Initialize TensorFlow.
    
    tflib.init_tf()
    random_state = np.random.RandomState(random_seed)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    args = sys.argv
    num_pngs = int(args[1]) if len(args)>1 else 10000
    num_gpus = int(args[2]) if len(args)>2 else 2
    out_dir  = args[3] if len(args)>3 else './genimg'
    pkl  = args[4] if len(args)>4 else 'networks/karras2019stylegan-celebahq-1024x1024.pkl'

    # Load pre-trained network.
    # with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    print('Loading network...')
    nowpath = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(nowpath, pkl), 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    print('Load network finished')
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
    # Print network details.
    # Gs.print_layers()
    # _D.print_layers()

    os.makedirs(out_dir, mode=0o777, exist_ok=True)
    latents = random_latents(num_pngs, Gs, random_state=random_state)
    labels = np.zeros([latents.shape[0], 0], np.float32)

    # images = Gs.run(latents, labels, minibatch_size=32, num_gpus=num_gpus, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    # print(f'images.shape: {images.shape}, {images.max()}, {images.min()}')
    # for png_idx in range(num_pngs):
        # print('Generating png to %s: %d / %d...' % (out_dir, png_idx, num_pngs), end='\r')
        # if not os.path.exists(os.path.join(out_dir, 'ProGAN_%08d.png' % png_idx)):
            # # misc.save_image_grid(images[png_idx:png_idx+1], os.path.join(out_dir, 'ProGAN_%08d.png' % png_idx), [0,255], [1,1])
            # PIL.Image.fromarray(images[png_idx], 'RGB').save(os.path.join(out_dir, 'StyleGAN_%08d.png' % png_idx))
    # print()
    
    
    png_idx = 1
    for latent, label in zip(latents,labels):
        latent, label = latent[np.newaxis,...], label[np.newaxis,...]
        images = Gs.run(latent, label, truncation_psi=0.7, num_gpus=num_gpus, randomize_noise=True, output_transform=fmt)
        # print(images.shape,images.max(),images.min());exit(0)
        # images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8).transpose(0, 2, 3, 1)
        print('Generating png to %s: %d / %d...' % (out_dir, png_idx, num_pngs),images[0].shape, end='\r')
        if not os.path.exists(os.path.join(out_dir, 'StyleGAN_%08d.png' % png_idx)):
            # misc.save_image_grid(images[png_idx:png_idx+1], os.path.join(out_dir, 'ProGAN_%08d.png' % png_idx), [0,255], [1,1])
            PIL.Image.fromarray(images[0], 'RGB').save(os.path.join(out_dir, 'StyleGAN_%08d.png' % png_idx))
            png_idx += 1
    print()





    # # Pick latent vector.
    # rnd = np.random.RandomState(5)
    # latents = rnd.randn(1, Gs.input_shape[1])

    # # Generate image.
    
    # images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # # Save image.
    # os.makedirs(config.result_dir, exist_ok=True)
    # png_filename = os.path.join(config.result_dir, 'example.png')
    # PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    print(sys.argv)
    main()
