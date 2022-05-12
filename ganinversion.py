# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10GhWnwIz5i8DVska6XtTOw6PSwJUIOqd
"""



"""# GAN Inversion

# Setup
"""

import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
pretrained_model_dir = os.path.join("/content", "models")
os.makedirs(pretrained_model_dir, exist_ok=True)

restyle_dir = os.path.join("/content", "restyle")
stylegan_ada_dir = os.path.join("/content", "stylegan_ada")
stylegan_nada_dir = os.path.join("/content", "stylegan_nada")

output_dir = os.path.join("/content", "output")

output_model_dir = os.path.join(output_dir, "models")
output_image_dir = os.path.join(output_dir, "images")

download_with_pydrive = True #@param {type:"boolean"}

class Downloader(object):
    def __init__(self, use_pydrive):
        self.use_pydrive = use_pydrive

        if self.use_pydrive:
            self.authenticate()
        
    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)
    
    def download_file(self, file_id, file_dst):
        if self.use_pydrive:
            downloaded = self.drive.CreateFile({'id':file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
        else:
            !gdown --id $file_id -O $file_dst

downloader = Downloader(download_with_pydrive)

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/yuval-alaluf/restyle-encoder.git $restyle_dir

!wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
!sudo unzip ninja-linux.zip -d /usr/local/bin/
!sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

!pip install ftfy regex tqdm 
!pip install git+https://github.com/openai/CLIP.git

!git clone https://github.com/NVlabs/stylegan2-ada/ $stylegan_ada_dir
!git clone https://github.com/rinongal/stylegan-nada.git $stylegan_nada_dir

from argparse import Namespace

import sys
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms

sys.path.append(restyle_dir)
sys.path.append(stylegan_nada_dir)
sys.path.append(os.path.join(stylegan_nada_dir, "ZSSGAN"))

device = 'cuda'

# %load_ext autoreload
# %autoreload 2

"""# Setup Restyle"""

from restyle.utils.common import tensor2im
from restyle.models.psp import pSp
from restyle.models.e4e import e4e

downloader.download_file("1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE", os.path.join(pretrained_model_dir, "restyle_psp_ffhq_encode.pt"))
downloader.download_file("1e2oXVeBPXMQoUoC_4TNwAWpOPpSEhE_e", os.path.join(pretrained_model_dir, "restyle_e4e_ffhq_encode.pt"))

"""# Choose a re-style model"""

encoder_type = 'e4e' #@param['psp', 'e4e']

restyle_experiment_args = {
    "model_path": os.path.join(pretrained_model_dir, f"restyle_{encoder_type}_ffhq_encode.pt"),
    "transform": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

model_path = restyle_experiment_args['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']

opts['checkpoint_path'] = model_path
opts = Namespace(**opts)

restyle_net = (pSp if encoder_type == 'psp' else e4e)(opts)

restyle_net.eval()
restyle_net.cuda()
print('Model successfully loaded!')

"""# Align and invert an image"""

def run_alignment(image_path):
    import dlib
    from scripts.align_faces_parallel import align_face
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

image_path = "/content/ariana.jpg" #@param {'type': 'string'} modify change to your own directory
original_image = Image.open(image_path).convert("RGB") 

input_image = run_alignment(image_path)

display(input_image)

img_transforms = restyle_experiment_args['transform']
transformed_image = img_transforms(input_image)

def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image

opts.n_iters_per_batch = 5
opts.resize_outputs = False  # generate outputs at full resolution

from restyle.utils.inference_utils import run_on_batch

with torch.no_grad():
    avg_image = get_avg_image(restyle_net)
    result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), restyle_net, opts, avg_image)

"""# Convert the image to the new domain"""

inverted_latent = torch.Tensor(result_latents[0][4]).cuda().unsqueeze(0).unsqueeze(1)

with torch.no_grad():
    net.eval()
    
    [sampled_src, sampled_dst] = net(inverted_latent, input_is_latent=True)[0]
    
    joined_img = torch.cat([sampled_src, sampled_dst], dim=0)
    save_images(joined_img, sample_dir, "joined", 2, 0)
    display(Image.open(os.path.join(sample_dir, f"joined_{str(0).zfill(6)}.jpg")).resize((512, 256)))