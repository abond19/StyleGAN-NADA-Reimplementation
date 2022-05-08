import os
import shutil

import torch
from torchvision import utils

import cv2

def save_images(images, output_dir, prefix, nrows, i):
  file_name=prefix+'_'+str(i).zfill(7)+'.jpg'
  utils.save_image(
        images,
        os.path.join(output_dir, file_name),
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )


def save_paper_image_grid(sampled_images, sample_dir, i):
    file_name=str(i)+'samples.png'
    img = (sampled_images + 1.0) * 126 

    half_size = img.size()[-1] // 2
    quarter_size = half_size // 2

    base_fig = torch.cat([img[0], img[1]], dim=2)
    sub_cols=torch.Tensor([])
    for i in range(2, 8, 2):
      for j in range(2):
        resized_img=torch.nn.functional.interpolate(img[i + j].unsqueeze(0), (half_size, half_size))[0]
        sub_cols=torch.cat(resized_img,dim=1)

    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    sub_cols=torch.Tensor([])
    for i in range(8, 16, 4):
      for j in range(4):
        resized_img=torch.nn.functional.interpolate(img[i + j].unsqueeze(0), (quarter_size, quarter_size))[0]
        sub_cols=torch.cat(resized_img,dim=1)

    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    save_torch_img(base_fig, sample_dir, file_name)


def save_torch_img(img, output_dir, file_name):
    img = img.permute(1, 2, 0).cpu().detach().numpy()

    img = img[:, :, ::-1]
    cv2.imwrite(os.path.join(output_dir, file_name), img)

def copytree(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        if os.path.isdir(s):
            copytree(s, os.path.join(dst, item))
        else:
            d=os.path.join(dst, item)
            if not os.path.exists(d):
                shutil.copy2(s, d)