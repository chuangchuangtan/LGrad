
# num_pngs = args[1] if len(args)>1 else 10000
# num_gpus = args[2] if len(args)>2 else 2
# out_dir  = args[3] if len(args)>3 else './genimg'
CUDA_VISIBLE_DEVICES=1 /usr/bin/python gen_example.py 10000 1 Celeba_HQ_StyleGAN/


# num_gpus  = args[1] if len(args)>1 else 1
# input_dir = args[2] if len(args)>1 else './genimg'
# out_dir   = args[3] if len(args)>1 else './genimg_grad'
CUDA_VISIBLE_DEVICES=1 /usr/bin/python gen_imggrad.py 1 Celeba_HQ_StyleGAN/ Celeba_HQ_StyleGAN_grad/