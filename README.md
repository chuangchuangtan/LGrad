# Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection

<p align="center">
	<br>
	Beijing Jiaotong University, YanShan University
</p>

<img src="./overall_pipeline.png" width="100%" alt="overall pipeline">

Reference github repository for the paper [Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection](url). Tan CC et al., proceedings of the IEEE/CVF CVPR 2023 . If you use our code, please cite our paper:
```
@inproceedings{tan2023learning,
title={Learning on Gradients: Generalized Artifacts Representation for {GAN}-Generated Images Detection},
author={Tan, Chuangchuang and Wei, Shikui and Gu, Guanghua and Wei, Yunchao and Zhao, Yao},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={}
}
```

## Environment setup
To start, we prefer creating the environment using conda:
```sh
conda env create -f environment.yml
conda activate lgrad
```

## Getting the data
Download dataset from [CNNDetection](https://github.com/peterwang512/CNNDetection).
## Transform Image to Gradients
1. Download pretrained model of [stylegan](https://github.com/NVlabs/stylegan), and put this `<project dir>/img2grad/stylegan/networks/`. Or run using
```sh
mkdir -p ./img2gad/stylegan/networks
wget https://lid-1302259812.cos.ap-nanjing.myqcloud.com/tmp/karras2019stylegan-bedrooms-256x256.pkl -O ./img2gad/stylegan/networks/karras2019styl
egan-bedrooms-256x256.pkl
```
2. Run using
```sh
sh ./transform_img2grad.sh {GPU-ID} {Data-Root-Dir} {Grad-Save-Dir}
```

## Training the model 
```sh
sh ./train-detector.sh {GPU-ID} {Grad-Save-Dir}
```

## Testing the detector
Download all pretrained weight files from<https://drive.google.com/drive/folders/17-MAyCpMqyn4b_DFP2LekrmIgRovwoix?usp=share_link>.
```sh
cd CNNDetection
CUDA_VISIBLE_DEVICES=0 python eval_test8gan.py --model_path {Model-Path}  --dataroot {Grad-Test-Path} --batch_size {BS}
```

## Acknowledgments

This repository borrows partially from the [CNNDetection](https://github.com/peterwang512/CNNDetection) and [stylegan](https://github.com/NVlabs/stylegan)
