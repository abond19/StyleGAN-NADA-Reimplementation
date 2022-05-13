import os
import sys
import torch
import shutil

from model.stylegannada_model import StyleGANNada
from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise
from utils.logger import log
from utils.argparser import parser, parse



"""if './hotel' not in sys.path:
    sys.path.insert(0, './hotel')
from hotel import *"""


def train(args, sample_dir, ckpt_dir, logger):
    net = StyleGANNada(args)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

    optimizer = torch.optim.Adam(net.trainable_generator.parameters(), lr=args.lr * g_reg_ratio,
                                 betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))

    noise = torch.randn(args.n_sample, 512, device=device)

    for i in range(args.iter):
        net.train()
        samples = mixing_noise(args.batch, 512, args.mixing, device)

        [sampled_src, sampled_dst], loss = net(samples)
        logger.info("iteration: {} loss: {:.6f}".format(i, loss))
        print(i, "th iteration ", loss)

        net.zero_grad()
        loss.backward()

        optimizer.step()

        if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net([noise], truncation=args.sample_truncation)

                # TODO:car cropping

                grid_rows = int(args.n_sample ** 0.5)

                save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

        if i % args.save_interval == 0:
            file_name = 'ckpt' + '_' + str(i).zfill(7) + '.pt'
            torch.save({
                'epoch': i,
                'model_state_dict': net.trainable_generator.generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(ckpt_dir, file_name))

    for i in range(args.num_grid_outputs):
        net.eval()
        with torch.no_grad():
            gen_sample = mixing_noise(16, 512, 0, device)
            [sampled_src, sampled_dst], loss = net(gen_sample, truncation=args.sample_truncation)

        # TODO: crop for cars

        save_paper_image_grid(sampled_dst, sample_dir, i)


if __name__ == "__main__":
    print(__package__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args, sample_dir, ckpt_dir = parse(parser)
    logger = log()
    code_dir = os.path.join(args.output_dir, "code")
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)

    criteria = os.path.join(args.output_dir, "code", "criteria")
    if not os.path.exists(criteria):
        os.makedirs(criteria)

    #copytree("model/criteria/", criteria)
    #shutil.copy2("model/Stylegannada_model.py", os.path.join(args.output_dir, "code", "Stylegannada_model.py"))

    train(args, sample_dir, ckpt_dir, logger)
