#!/bin/bash

pwd=$(cd $(dirname $0); pwd)
cd ${pwd}/CNNDetection/


CUDA_VISIBLE_DEVICES=$1 python train.py --name 1class-resnet-horse --dataroot ${pwd}/img2gad/stylegan/ --classes horse --batch_size 16 --delr_freq 10 --lr 0.0005 --niter 100

CUDA_VISIBLE_DEVICES=$1 python train.py --name 2class-resnet-chair-horse --dataroot ${pwd}/img2gad/stylegan/ --classes chair,horse --batch_size 16 --delr_freq 10 --lr 0.0005 --niter 100

CUDA_VISIBLE_DEVICES=$1 python train.py --name 4class-resnet-car-cat-chair-horse --dataroot ${pwd}/img2gad/stylegan/ --classes car,cat,chair,horse --batch_size 16 --delr_freq 10 --lr 0.0005 --niter 100

