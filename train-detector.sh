#!/bin/bash

pwd=$(cd $(dirname $0); pwd)
cd ${pwd}/CNNDetection/

#${pwd}/img2gad/stylegan/Grad_dataset

CUDA_VISIBLE_DEVICES=$1 python train.py --name 1class-resnet-horse --dataroot $2 --classes horse --batch_size 16 --delr_freq 10 --lr 0.0005 --niter 100

CUDA_VISIBLE_DEVICES=$1 python train.py --name 2class-resnet-chair-horse --dataroot $2 --classes chair,horse --batch_size 16 --delr_freq 10 --lr 0.0005 --niter 100

CUDA_VISIBLE_DEVICES=$1 python train.py --name 4class-resnet-car-cat-chair-horse --dataroot $2 --classes car,cat,chair,horse --batch_size 16 --delr_freq 10 --lr 0.0005 --niter 100

