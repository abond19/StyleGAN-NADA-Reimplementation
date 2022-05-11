import argparse
import os

parser = argparse.ArgumentParser(description='StyleGAN-NADA arguments')

parser.add_argument(
    "--frozen_gen_ckpt",
    type=str,
    help="Path to a pre-trained StyleGAN2 generator for use as the initial frozen network. ",
    required=True
)

parser.add_argument(
    "--train_gen_ckpt",
    type=str,
    help="Path to a pre-trained StyleGAN2 generator for use as the initial trainable network.",
    required = True,
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
)

parser.add_argument(
    "--lambda_direction",
    type=float,
    default=1.0,
    help="Strength of directional clip loss",
)

parser.add_argument(
    "--save_interval",
    type=int,
    help="checkpoints save interval",
)

parser.add_argument(
    "--output_interval",
    type=int,
    default=100,
    help="output save interval(image)",
)

parser.add_argument(
    "--source_class",
    default="dog",
)

parser.add_argument(
    "--target_class",
    default="cat",
)

parser.add_argument(
    "--sample_truncation",
    default=0.7,
    type=float,
    help="test images truncation"
)

parser.add_argument(
    "--auto_layer_iters",
    type=int,
    default=1,
)

parser.add_argument(
    "--auto_layer_k",
    type=int,
    default=1,
)

parser.add_argument(
    "--auto_layer_batch",
    type=int,
    default=8,
    help="Batch size for use in automatic layer selection step."
)

parser.add_argument(
    "--clip_models",
    nargs="+",
    type=str,
    default=["ViT-B/32"],
    help="Names of CLIP models to use for losses"
)

parser.add_argument(
    "--clip_model_weights",
    nargs="+",
    type=float,
    default=[1.0],
    help="Relative loss weights of the clip models"
)

parser.add_argument(
    "--num_grid_outputs",
    type=int,
    default=0,
    help="Number of paper-style grid images to generate after training."
)

# car aspect ratio

# style targets
parser.add_argument(
    "--style_img_dir",
    type=str,
)

parser.add_argument(
    "--img2img_batch",
    type=int,
    default=16,
)

# Original rosinality args.

parser.add_argument(
    "--iter", type=int, default=1000,
)
parser.add_argument(
    "--batch", type=int, default=16,
)

parser.add_argument(
    "--n_sample",
    type=int,
    default=64,
)

parser.add_argument(
    "--size", type=int, default=256, help="image size"
)

parser.add_argument(
    "--r1", type=float, default=10, help="r1 regularization"
)

parser.add_argument(
    "--path_regularize",
    type=float,
    default=2,
    help="weight of the path length regularization",
)

parser.add_argument(
    "--path_batch_shrink",
    type=int,
    default=2,
    help="batch size reducing factor for the path length regularization (reduce memory consumption)",
)

parser.add_argument(
    "--d_reg_every",
    type=int,
    default=16,
    help="interval of the applying r1 regularization",
)

parser.add_argument(
    "--g_reg_every",
    type=int,
    default=4,
    help="interval of the applying path length regularization",
)

parser.add_argument(
    "--mixing", type=float, default=0.0, help="probability of latent code mixing"
)

parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help="path to the checkpoints to resume training",
)

parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

parser.add_argument(
    "--channel_multiplier",
    type=int,
    default=2,
    help="channel multiplier factor for the model. config-f = 2, else = 1",
)

parser.add_argument(
    "--augment", action="store_true", help="apply non leaking augmentation"
)

parser.add_argument(
    "--augment_p",
    type=float,
    default=0,
    help="probability of applying augmentation. 0 = use adaptive augmentation",
)

parser.add_argument(
    "--ada_target",
    type=float,
    default=0.6,
    help="target augmentation probability for adaptive augmentation",
)

parser.add_argument(
    "--ada_length",
    type=int,
    default=500 * 1000,
    help="target duraing to reach augmentation probability for adaptive augmentation",
)

parser.add_argument(
    "--ada_every",
    type=int,
    default=256,
    help="probability update interval of the adaptive augmentation",
)


def parse(arg_parser):
    args = arg_parser.parse_args()
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir = os.path.join(args.output_dir, "checkpoint")

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.style_img_dir:
        filelist = os.listdir(args.style_img_dir)
        for image in filelist[:]:
            if not (image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg")):
                filelist.remove(image)

        args.style_img_list = filelist

    return args, sample_dir, ckpt_dir
