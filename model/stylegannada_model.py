from stylegan2_model import Generator, Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import torchvision.transforms as transforms

def requires_grad(layers, flag=True):
    for p in layers.parameters():
        p.requires_grad = flag

class NADAGenerator(nn.Module):
    
    def __init__(self, checkpoint_path="stylegan2-ffhq-config-f.pt", style_dims=512, layers=8, image_size=1024, channel_multiplier=2, device="cuda:0"):
        super(NADAGenerator, self).__init__()
        self.latent_size = style_dims
        self.layers = layers
        self.image_size = image_size
        self.channel_multiplier = channel_multiplier
        
        self.generator = Generator(image_size, style_dims, layers, channel_multiplier)
        
        checkpoint_file = torch.load("stylegan2-ffhq-config-f.pt")
        
        self.generator.load_state_dict(checkpoint_file["g_ema"], strict=False)
        
    def forward(self, styles, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_is_latent=False, input_is_s_code=False, noise=None, randomize_noise=True):
        return self.generator(styles, return_latents=return_latents, inject_index=inject_index, truncation=truncation, truncation_latent=truncation_latent, input_is_latent=input_is_latent, input_is_s_code=input_is_s_code, noise=noise, randomize_noise=randomize_noise)
    
    def get_training_layers(self, phase):

        if phase == 'texture':
            result = list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
        if phase == 'shape':
             result = list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
             result = list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
             result = list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            result = self.get_all_layers() 
        else: 
            result = list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:]) 
        
        return result
        
    def get_all_layers(self):
        return list(self.generator.children())
    
    def freeze_layers(self, layer_list=None):
        if layer_list == None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layers:
                requires_grad(layer, False)
    
    def unfreeze_layers(self, layer_list=None):
        if layer_list == None:
            self.unfreeze_layers(self.get_all_layers)
        else:
            for layer in layer_list:
                requires_grad(layer, True)
                
    def style(self, styles):
        styles = [self.generator.style(s) for s in styles]
        return styles
    
    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)
    
    def modulation_layers(self):
        return self.generator.modulation_layers
    
    

class NADADiscriminator(nn.Module):
    
    def __init__(self, checkpoint_path="stylegan2-ffhq-config-f.pt", style_dims=512, layers=8, image_size=1024, channel_multiplier=2, device="cuda:0"):
        super(NADADiscriminator, self).__init__()
        self.latent_size = style_dims
        self.layers = layers
        self.image_size = image_size
        self.channel_multiplier = channel_multiplier
        
        self.discriminator = Discriminator(image_size, channel_multiplier)
        
        checkpoint_file = torch.load("stylegan2-ffhq-config-f.pt")
        
        self.discriminator.load_state_dict(checkpoint_file["d"], strict=False)
    
    def forward(self, x):
        return self.discriminator(x)
    
    def get_all_layers(self):
        return list(self.discriminator.children())
    
    def get_training_layers(self):
        return self.get_all_layers()
    
    def freeze_layers(self, layer_list=None):
        if layer_list == None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layers:
                requires_grad(layer, False)
    
    def unfreeze_layers(self, layer_list=None):
        if layer_list == None:
            self.unfreeze_layers(self.get_all_layers)
        else:
            for layer in layer_list:
                requires_grad(layer, True)
        
        
class StyleGANNada(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        self.device = "cuda:0"
        
        # Need both generators for the training loss
        self.frozen_generator = NADAGenerator(args.frozen_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier).to(self.device)
        self.frozen_generator.freeze_layers()
        self.frozen_generator.eval()
        
        self.trainable_generator = NADAGenerator(args.train_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier).to(self.device)
        self.trainable_generator.freeze_layers()
        self.trainable_generator.unfreeze_layers(self.trainable_generator.get_training_layers(args.phase))
        self.trainable_generator.train()
        
        self.mse_loss = nn.MSELoss()
        
        # Need to know the current and the target class, so we can move in the correct direction
        self.source_class = args.source_class
        self.target_class = args.target_class
        
        self.auto_layer_k = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters
        
        # Need the clip model in order to calculate the loss
        self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}
        self.clip_loss_models = {model_name: CLIPLoss(self.device, lambda_direction=args.lambda_direction, lambda_patch=args.lambda_patch, lambda_global=args.lambda_global, lambda_manifold=args.lambda_manifold, lambda_texture=args.lambda_texture, clip_model=model_name) for model_name in args.clip_models}

        if args.target_img_list is not None:
            self.set_img2img_direction()
    
    # Adjust the layers of the frozen generator to have the same weights as the learning generator
    def pivot(self):
        frozen_layers = dict(self.frozen_generator.named_parameters())
        train_layers = dict(self.trainable_generator.named_parameters())
        
        for k in frozen_layers.keys():
            frozen_layers[k] = train_layers[k]
    
    def set_img2img_directions(self):
        with torch.no_grad():
            random_value = torch.randn(self.args.img2img_batch, 512, device=self.device)
            generated = self.trainable_generator([random_value])[0]
            
            for _, model in self.clip_loss_models.items():
                direction = model.compute_img2img_direction(generated, self.args.target_img_list)
                
                model.target_direction = direction
    
    def determine_optimal_layers(self):
        random_value = torch.randn(self.args.auto_layer_batch, 512, device=self.device)
        
        initial_w = self.frozen_generator.style([sample_z])
        initial_w = initial_w[0].unsqueeze(1).repeat(1, self.frozen_generator.generator.n_latent, 1)
        
        w = torch.Tensor(initial_w.cpu().detach().numpy()).to(self.device)
        
        w.requires_grad = True
        
        w_optim = optim.SGD([w], lr=0.01)
        
        for _ in range(self.auto_layer_iters):
            w_gen = w.unsqueeze(0)
            generated_from_w = self.trainable_generator(w_gen, input_is_latent=True)[0]
            
            w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_class) for model_name in self.clip_model_weights.keys()]
            
            w_loss = torch.sum(torch.stack(w_loss))
            
            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()
            
        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        chosen_layers = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()
        
        all_layers = list(self.trainable_generator.get_all_layers())
        
        conv_layers = list(all_layers[4])
        
        rgb_layers = list(all_layers[6])
        
        idx_to_layers = all_layers[2:4] + conv_layers
        
        chosen_layers = [idx_to_layers[idx] for idx in chosen_layers]
        
        return chosen_layers
    
    