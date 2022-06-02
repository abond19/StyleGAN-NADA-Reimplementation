import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os
import math
import clip
from PIL import Image
from .text_templates import imagenet_templates, part_templates

class DirectionLoss(torch.nn.Module):
  def  __init__(self, loss_type='mse'):
    super(DirectionLoss, self).__init__()
    self.loss_type=loss_type
    
    if self.loss_type=='mse':
      self.loss_f=nn.MSELoss()
    elif self.loss_type=='cosine':
      self.loss_f=nn.CosineSimilarity()
    elif self.loss_type=='mae':
      self.loss_f=nn.L1Loss()


  def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_f(x, y)
        
        return self.loss_f(x, y)

class CLIPLoss(torch.nn.Module):
  def __init__(self, device, lambda_direction=1., lambda_patch=0., lambda_global=0., lambda_manifold=0., lambda_texture=0., patch_loss_type='mae', direction_loss_type='cosine', clip_model='ViT-B/32'):
      super(CLIPLoss, self).__init__()
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.model, clip_preprocess = clip.load(clip_model, device=self.device)
      self.clip_preprocess = clip_preprocess
      
      self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                            clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                            clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor
      self.target_direction      = None
      self.patch_text_directions = None
      self.patch_loss     = DirectionLoss(patch_loss_type)
      self.direction_loss = DirectionLoss(direction_loss_type)
      self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)
      self.lambda_global    = lambda_global
      self.lambda_patch     = lambda_patch
      self.lambda_direction = lambda_direction
      self.lambda_manifold  = lambda_manifold
      self.lambda_texture   = lambda_texture
      self.src_text_features = None
      self.target_text_features = None
      self.angle_loss = torch.nn.L1Loss()
      self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
      self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                      preprocess_cnn.transforms[:2] +                                                 # to match CLIP input scale assumptions
                                      preprocess_cnn.transforms[4:])                                                  # + skip convert PIL to tensor
      self.texture_loss = torch.nn.MSELoss()
  
  def compute_text_direction(self, source_class, target_class):
      template_text =  [template.format(source_class) for template in imagenet_templates]
      tokens = clip.tokenize(template_text).to(self.device)

      source_features = self.model.encode_text(tokens).detach()
      source_features /= source_features.norm(dim=-1, keepdim=True)

      template_text =  [template.format(target_class) for template in imagenet_templates]
      tokens = clip.tokenize(template_text).to(self.device)

      target_features = self.model.encode_text(tokens).detach()
      target_features /= source_features.norm(dim=-1, keepdim=True)
       

      text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
      text_direction /= text_direction.norm(dim=-1, keepdim=True)

      return text_direction
  
  def compute_img2img_direction(self, source_images, target_images):
    with torch.no_grad():

        src_encoding = self.encode_images(source_images)
        src_encoding /= src_encoding.clone().norm(dim=-1, keepdim=True)
        src_encoding = src_encoding.mean(dim=0, keepdim=True)

        target_encodings = []
        for target_img in target_images:
            preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                
            encoding = self.model.encode_image(preprocessed)
            encoding /= encoding.norm(dim=-1, keepdim=True)

            target_encodings.append(encoding)
            
        target_encoding = torch.cat(target_encodings, axis=0)
        target_encoding = target_encoding.mean(dim=0, keepdim=True)

        direction = target_encoding - src_encoding
        direction /= direction.norm(dim=-1, keepdim=True)

    return direction

  def clip_angle_loss(self, src_img, source_class, target_img, target_class):
      if self.src_text_features is None:
          source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
          self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

          target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
          self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

      cos_text_angle = self.target_text_features @ self.src_text_features.T
      text_angle = torch.acos(cos_text_angle)

      src_img_features = self.encode_images(src_img)
      src_img_features /= src_img_features.clone().norm(dim=-1, keepdim=True).unsqueeze(2)
      target_img_features = self.encode_images(target_img)
      target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True).unsqueeze(1)

      cos_img_angle = torch.clamp(target_img_features @ src_img_features, min=-1.0, max=1.0)
      img_angle = torch.acos(cos_img_angle)

      text_angle = text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)
      cos_text_angle = cos_text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)

      return self.angle_loss(cos_img_angle, cos_text_angle)
  
  def clip_directional_loss(self, src_img, source_class, target_img, target_class):

      if self.target_direction is None:
        template_text = [template.format(source_class) for template in imagenet_templates] 
        tokens = clip.tokenize(template_text).to(self.device)
        source_features = self.model.encode_text(tokens).detach()
        source_features /= source_features.norm(dim=-1, keepdim=True)

        template_text = [template.format(target_class) for template in imagenet_templates] 
        tokens = clip.tokenize(template_text).to(self.device)
        target_features = self.model.encode_text(tokens).detach()
        target_features /= target_features.norm(dim=-1, keepdim=True)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        self.target_direction = text_direction

      images = self.preprocess(src_img).to(self.device)
      src_encoding= self.model.encode_image(images)
      src_encoding /= src_encoding.clone().norm(dim=-1, keepdim=True)
 
      image = self.preprocess(target_img).to(self.device)
      target_encoding= self.model.encode_image(image)
      target_encoding /= target_encoding.clone().norm(dim=-1, keepdim=True)

      edit_direction = (target_encoding - src_encoding)
      edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        
      return self.direction_loss(edit_direction, self.target_direction).mean()

  def global_clip_loss(self, img, text):
      if not isinstance(text, list):
          text = [text]
            
      tokens = clip.tokenize(text).to(self.device)
      image  = self.preprocess(img)

      logits_per_image, _ = self.model(image, tokens)

      return (1. - logits_per_image / 100).mean()
  
  def create_patch(self,img,patch_centers, batch_idx,patch_idx, num_patches,size):
    center_x = patch_centers[batch_idx * num_patches + patch_idx][0]
    center_y = patch_centers[batch_idx * num_patches + patch_idx][1]
    
    return img[batch_idx:batch_idx+1, :, center_y - (size // 2):center_y + (size // 2), center_x - (size // 2):center_x + (size // 2)]

  def generate_patches(self, img: torch.Tensor, patch_centers, size):
      batch_size  = img.shape[0]
      num_patches = len(patch_centers) // batch_size
      patches = torch.cat([self.create_patch(img,patch_centers, batch_idx,patch_idx, num_patches,size) for patch_idx in range(num_patches) for batch_idx in range(batch_size)], dim=0)

      return patches
  
  def patch_scores(self, img, class_str, patch_centers, patch_size):

      parts = [template.format(class_str) for template in part_templates]
      tokens = clip.tokenize(parts).to(self.device)
      text_features = self.model.encode_text(tokens).detach()

      batch_size  = img.shape[0]
      num_patches = len(patch_centers) // batch_size
      patches = torch.cat([self.create_patch(img,patch_centers, batch_idx,patch_idx, num_patches,patch_size) for patch_idx in range(num_patches) for batch_idx in range(batch_size)], dim=0)
      image_features = self.encode_images(patches)
      image_features /= image_features.clone().norm(dim=-1, keepdim=True)

      similarity = torch.matmul(image_features,text_features.T)

      return similarity

  def clip_patch_similarity(self, src_img, source_class, target_img, target_class):
      # 98, 4 and 196 are magical numbers taken from code
      batch_size, channels, height, width = src_img.shape
      patch_centers = np.concatenate([np.random.randint(98, width - 98,  size=(batch_size * 4, 1)),
                                        np.random.randint(98, height - 98, size=(batch_size * 4, 1))], axis=1)

       
   
      src_scores    = self.patch_scores(src_img, source_class, patch_centers, 196)
      target_scores = self.patch_scores(target_img, target_class, patch_centers, 196)

      return self.patch_loss(src_scores, target_scores)
  
  def patch_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

      if self.patch_text_directions is None:
          src_part_classes = [template.format(source_class) for template in part_templates]
          target_part_classes = [template.format(target_class) for template in part_templates] 

          parts_classes = list(zip(src_part_classes, target_part_classes))

          self.patch_text_directions = torch.cat([self.compute_text_direction(pair[0], pair[1]) for pair in parts_classes], dim=0)

      #again a magical number from official code
      patch_size = 510
      batch_size, channels, height, width = src_img.shape
      patch_centers = np.concatenate([np.random.randint(255, width - 255,  size=(batch_size * 1, 1)),
                                        np.random.randint(255, height - 255, size=(batch_size * 1, 1))], axis=1)

      patches = self.generate_patches(src_img, patch_centers, patch_size)
      src_features = self.encode_images(patches)
      src_features /= src_features.clone().norm(dim=-1, keepdim=True)

      patches = self.generate_patches(target_img, patch_centers, patch_size)
      target_features = self.encode_images(patches)
      target_features /= target_features.clone().norm(dim=-1, keepdim=True)

      edit_direction = (target_features - src_features)
      edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

      cosine_dists = 1. - self.patch_direction_loss(edit_direction.unsqueeze(1), self.patch_text_directions.unsqueeze(0))

      patch_class_scores = cosine_dists * (torch.matmul(edit_direction, self.patch_text_directions.T)).softmax(dim=-1)

      return patch_class_scores.mean()
  
  def forward(self, src_img, source_class, target_img, target_class, texture_image= None):
      clip_loss = 0.0

      if self.lambda_global:
          clip_loss += self.lambda_global * self.global_clip_loss(target_img, [f"a {target_class}"])
          print(1)
          print(clip_loss)

      if self.lambda_patch:
          clip_loss += self.lambda_patch * self.patch_directional_loss(src_img, source_class, target_img, target_class)
          print(2)
          print(clip_loss)

      if self.lambda_direction:
          clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img, target_class)


      if self.lambda_manifold:
          clip_loss += self.lambda_manifold * self.clip_angle_loss(src_img, source_class, target_img, target_class)
          print(4)
          print(clip_loss)

      if self.lambda_texture and (texture_image is not None):
          texture_image = self.preprocess_cnn(texture_image).to(self.device)
          src_features= self.model_cnn.encode_image(texture_image)
       
          target_img = self.preprocess_cnn(target_img).to(self.device)
          target_features= self.model_cnn.encode_image(target_img)
          cnn_feature_loss=self.texture_loss(src_features, target_features)

          clip_loss += self.lambda_texture * cnn_feature_loss
          print(5)
          print(clip_loss)

      return clip_loss