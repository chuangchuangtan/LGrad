#!/bin/bash

#./transform_img2grad.sh 0 /root/ img2gad/stylegan/Grad_dataset/
Classes='0_real 1_fake'
GANmodelpath=$(cd $(dirname $0); pwd)/img2gad/stylegan/
Model=karras2019stylegan-bedrooms-256x256.pkl
Imgrootdir=$2
Saverootdir=$3


# Valdatas='airplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'
Valdatas='horse car cat chair'
Valrootdir=${Imgrootdir}/val/
Savedir=$Saverootdir/val/

for Valdata in $Valdatas
do
    for Class in $Classes
    do
        Imgdir=${Valdata}/${Class}
        CUDA_VISIBLE_DEVICES=$1 /usr/bin/python $GANmodelpath/img2grad.py 1\
            ${Valrootdir}${Imgdir} \
            ${Savedir}${Imgdir}_grad \
            ${GANmodelpath}networks/${Model} \
            1
    done
done


Traindatas='horse car cat chair'
Trainrootdir=${Imgrootdir}/train/
Savedir=$Saverootdir/train/
for Traindata in $Traindatas
do
    for Class in $Classes
    do
        Imgdir=${Traindata}/${Class}
        CUDA_VISIBLE_DEVICES=$1 /usr/bin/python $GANmodelpath/img2grad.py 1\
            ${Trainrootdir}${Imgdir} \
            ${Savedir}${Imgdir}_grad \
            ${GANmodelpath}networks/${Model} \
            1
    done
done


Testdatas='biggan deepfake gaugan stargan cyclegan/apple cyclegan/horse cyclegan/orange cyclegan/summer cyclegan/winter cyclegan/zebra progan/airplane progan/bicycle progan/bird progan/boat progan/bottle progan/bus progan/car progan/cat progan/chair progan/cow progan/diningtable progan/dog progan/horse progan/motorbike progan/person progan/pottedplant progan/sheep progan/sofa progan/train progan/tvmonitor stylegan/bedroom stylegan/car stylegan/cat stylegan2/car stylegan2/cat stylegan2/church stylegan2/horse'
Testrootdir=${Imgrootdir}/test/
Savedir=$Saverootdir/test/

for Testdata in $Testdatas
do
    for Class in $Classes
    do
        Imgdir=${Testdata}/${Class}
        CUDA_VISIBLE_DEVICES=$1 /usr/bin/python $GANmodelpath/img2grad.py 1\
            ${Testrootdir}${Imgdir} \
            ${Savedir}${Imgdir}_grad \
            ${GANmodelpath}networks/${Model} \
            1
    done
done

