# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from datetime import datetime, timedelta

import cv2
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.utils as torch_utils
from tqdm import tqdm

from mtcnn_periocular_crop import crop_image
import legacy

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    G.eval()

    os.makedirs(outdir, exist_ok=True)

    seeds = 99836
    batchsize = 16
    
    np.random.seed(1337)
    
    random_data = np.zeros((seeds, G.z_dim))
    print("Generating random matrix")
    for i in tqdm(range(len(random_data))):
        random_data[i] = np.random.RandomState(i).randn(G.z_dim)
    print("Input shape:")
    print(random_data.shape)

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    num_batches = (seeds // batchsize) + 1
    ba = 0
    img_num = 0
    batch_num = 1
    # Generate images.
    start_time = datetime.now()
    
    with torch.no_grad():
        while ba < len(random_data):
            bb = min(ba + batchsize, len(random_data))
            _data = random_data[bb - batchsize: bb]
            z = torch.from_numpy(_data).to(device)
            imgs = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            imgs = F.interpolate(imgs, size=(224,224))
            imgs = imgs.detach().cpu() # .numpy()
            for img in imgs:
                img = img - img.min()
                img = img / (img.max() - img.min())
                torch_utils.save_image(img, f'{outdir}/img{img_num:06d}.png')
                img_num += 1

            curr = datetime.now() - start_time
            secs_per_batch = curr.seconds / batch_num
            remaining = (num_batches-batch_num) * secs_per_batch
            print(f"batch {batch_num}/{num_batches}", "{:0>8} to go".format(str(timedelta(seconds=remaining))))
            ba = bb
            batch_num += 1



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------