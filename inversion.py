import os
from argparse import Namespace
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms



from restyle.utils.common import tensor2im
from restyle.models.psp import pSp
from restyle.models.e4e import e4e
from restyle.scripts.align_faces_parallel import align_face
from restyle.utils.inference_utils import run_on_batch


def restyle_experiment_arguments(pretrained_model_dir,encoder_type):
    restyle_experiment_args = {
        "model_path": os.path.join('/kuacc/users/yakarken18/StyleGAN-NADA-Reimplementation', f"restyle_e4e_ffhq_encode.pt"),
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
    return restyle_experiment_args

def run_alignment(image_path):
    import dlib
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    #print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    return avg_image