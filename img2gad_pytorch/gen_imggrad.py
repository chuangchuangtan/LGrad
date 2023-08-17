
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from glob import glob
import cv2
import sys
import os
import PIL.Image
import torch.nn.functional as F
from torchvision import transforms
from models import build_model



processimg = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
#            transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
        ])


def read_batchimg(imgpath_list):
    img_list = []
    for imgpath in imgpath_list:
        img_list.append( torch.unsqueeze(processimg(PIL.Image.open(imgpath).convert('RGB')),0)  )
    return torch.cat(img_list,0)
    
def normlize_np(img):
    img -= img.min()
    if img.max()!=0: img /= img.max()
    return img * 255.                

def get_imglist(path):
    ext = [".jpg",".bmp",".png",".jpeg",".webp"]   # Add image formats here
    files = []
    [files.extend(glob(os.path.join(path, f'*{e}'))) for e in ext]
    return sorted(files)

def generate_images():
    imgdir = sys.argv[1]
    outdir = sys.argv[2]
    modelpath = sys.argv[3]
    batch_size = int(sys.argv[4]) if len(sys.argv)>4 else 1
    print(f'Transform {imgdir} to {outdir}')
    os.makedirs(outdir, exist_ok=True)

    model = build_model(gan_type='stylegan',
        module='discriminator',
        resolution=256,
        label_size=0,
        # minibatch_std_group_size = 1,
        image_channels=3)
    model.load_state_dict(torch.load(modelpath), strict=True)

    model.cuda()
    model.eval()

    imgnames_list = get_imglist(imgdir)
    if len(imgnames_list) == 0: exit()
    num_items = len(imgnames_list)
    print(f'From {imgdir} read {num_items} Img')
    minibatch_size = int(batch_size)
    numnow = 0
    for mb_begin in range(0, num_items, minibatch_size):
        imgname_list = imgnames_list[mb_begin: min(mb_begin+minibatch_size,num_items)]
        imgs_np      = read_batchimg(imgname_list)
        tmpminibatch = len(imgname_list)
        img_cuda = imgs_np.cuda().to(torch.float32)
        img_cuda.requires_grad = True
        pre = model(img_cuda)
        model.zero_grad()
        grad = torch.autograd.grad(pre.sum(), img_cuda, create_graph=True, retain_graph=True, allow_unused=False)[0]
        for idx in range(tmpminibatch):
            numnow += 1
            img = normlize_np(grad[idx].permute(1,2,0).cpu().detach().numpy())
            print(f'Gen grad to {os.path.join(outdir, imgname_list[idx].split("/")[-1])}, bs:{minibatch_size} {numnow}/{num_items}',end='\r')
            cv2.imwrite(os.path.join(outdir, imgname_list[idx].split('/')[-1].split('.')[0]+'.png'),img[...,::-1])
    print()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
