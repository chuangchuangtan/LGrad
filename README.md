# Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection

<p align="center">
	<br>
	BeijingJiaotong University
</p>


Reference github repository for the paper [Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection](url). Tan CC et al., proceedings of the IEEE/CVF CVPR 2023 . If you use our code, please cite our paper:
```
@inproceedings{
tancc2023learning,
title={Learning on Gradients: Generalized Artifacts Representation for {GAN}-Generated Images Detection},
author={tancc},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={https://openreview.net/forum?id=MRZb82xUbi}
}
```

## Environment setup
To start, we prefer creating the environment using conda:
```sh
conda env create -f environment.yml
conda activate lgrad
```

## Getting the data

## Transform Image to Gradients
1. Download pretrained model of [stylegan](https://github.com/NVlabs/stylegan), and put this `<project dir>/img2grad/stylegan/network/`.
2. Run using
```sh
sh ./transform_img2grad.sh {Data-Root-Dir} {Grad-Save-Dir}
```

## Training the model 
```sh
sh train-detector.sh
```

## Testing the detector
Download all pretrained weight files from<https://drive.google.com/drive/folders/17-MAyCpMqyn4b_DFP2LekrmIgRovwoix?usp=share_link>.
```sh
cd CNNDetection
CUDA_VISIBLE_DEVICES=0 python eval8gan.py --model_path {Model-Path}  --dataroot {Grad-Test-Path}
```

## Acknowledgments

This repository borrows partially from the [CNNDetection](https://github.com/peterwang512/CNNDetection) and [stylegan](https://github.com/NVlabs/stylegan)

<!-- #### Other Training Options
* <i>--patch_size</i>: training patch size
* <i>--img_mini_b</i>: image mini-batch size
* <i>--epoch</i>: number of training epochs
* <i>--lr</i>: initial learning rate
* <i>--schedule_lr_rate</i>: learning rate scheduler (after how many epochs to decrease)
* <i>--bit_depth</i>: image bit depth datatype, 16 for `uint16` or 8 for `uint8`. Recall that we train with 16-bit images
* <i>--dropout_rate</i>: the dropout rate of the `conv` unit at the network bottleneck
 -->
