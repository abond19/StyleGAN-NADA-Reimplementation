import os
import shutil

import torch
from torchvision import utils
import torch.nn.functional as F

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
    sub_cols = []
    for i in range(2, 8, 2):
        resized_img = F.interpolate(img[i].unsqueeze(0), (half_size, half_size))[0]
        resized_img1 = F.interpolate(img[i + 1].unsqueeze(0), (half_size, half_size))[0]

        sub_cols.append(torch.cat((resized_img, resized_img1), dim=1))

    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    sub_cols = []
    for i in range(8, 16, 4):
        resized_img = F.interpolate(img[i].unsqueeze(0), (quarter_size, quarter_size))[0]
        resized_img1 = F.interpolate(img[i + 1].unsqueeze(0), (quarter_size, quarter_size))[0]
        resized_img2 = F.interpolate(img[i + 2].unsqueeze(0), (quarter_size, quarter_size))[0]
        resized_img3 = F.interpolate(img[i + 3].unsqueeze(0), (quarter_size, quarter_size))[0]
        sub_cols.append(torch.cat((resized_img, resized_img1, resized_img2, resized_img3), dim=1))

    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    save_torch_img(base_fig, sample_dir, file_name)


def save_torch_img(img, output_dir, file_name):
    img = img.permute(1, 2, 0).cpu().detach().numpy()

    img = img[:, :, ::-1]
    cv2.imwrite(os.path.join(output_dir, file_name), img)

