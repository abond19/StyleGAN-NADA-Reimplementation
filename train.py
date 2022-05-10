import argparse
import os
import numpy as np

import torch


from model.ZSSGAN import ZSSGAN

import shutil
import json

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise
from utils.logger import log

args={
'size': 1024,
'batch': 2, 
'n_sample': 4, 
'output_dir': 'output_dir' ,
'lr': 0.002 ,
'frozen_gen_ckpt': '/path/to/stylegan2-ffhq-config-f.pt',
'iter': 301 ,
'source_class': "photo" ,
'target_class': "sketch", 
'lambda_direction': 1.0 ,
'lambda_patch': 0.0 ,
'lambda_global': 0.0 ,
'lambda_texture': 0.0 ,
'lambda_manifold': 0.0 ,
'phase': None ,
'auto_layer_k': 0 ,
'auto_layer_iters': 0 ,
'auto_layer_batch': 8 ,
'output_interval': 50 ,
'clip_models': "ViT-B/32" "ViT-B/16" ,
'clip_model_weights': 1.0 ,
'mixing': 0.0,
'save_interval': 50,
}

args = argparse.Namespace(**args)

save_dst=True
save_src=False


sample_dir = os.path.join(args.output_dir, "sample")
ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

logger = log()

def train(args):

    net=ZSSGAN(args)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) 

    optimizer = torch.optim.Adam(net.generator_trainable.parameters(),lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))

    noise= torch.randn(args.n_sample, 512, device=device)

    for i in range(args.iter):
      net.train()
      samples=mixing_noise(args.batch,512,args.mixing,device)

      [sampled_src, sampled_dst], loss = net(samples)
      logger.info("iteration: {} loss: {:.6f}".format(i, loss))


      net.zero_grad()
      loss.backward()

      optimizer.step()

      if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net([noise], truncation=args.sample_truncation)

                #TODO:car cropping

                grid_rows = int(args.n_sample ** 0.5)

                if save_src:
                    save_images(sampled_src, sample_dir, "src", grid_rows, i)

                if save_dst:
                    save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

      if (i % args.save_interval == 0):
        file_name='ckpt'+'_'+str(i).zfill(7)+'.pt'
        torch.save({
            'epoch': i,
            'model_state_dict': net.generator_trainable.generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(ckpt_dir, file_name))
        
    for i in range(args.num_grid_outputs):
        net.eval()
        with torch.no_grad():
            gen_sample = mixing_noise(16, 512, 0, device)
            [sampled_src, sampled_dst], loss = net(gen_sample, truncation=args.sample_truncation)

           #TODO: crop for cars

        save_paper_image_grid(sampled_dst, sample_dir, i)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    code_dir   = os.path.join(args.output_dir, "code")
    if not os.path.exists(code_dir):
      os.makedirs(code_dir)

    criteria   = os.path.join(args.output_dir, "code", "criteria")
    if not os.path.exists(criteria):
      os.makedirs(criteria)
    
    copytree("criteria/", criteria)
    shutil.copy2("model/ZSSGAN.py", os.path.join(args.output_dir, "code", "ZSSGAN.py"))
    
    
    train(args)

