import os
from argparse import ArgumentParser, Namespace
import sys
from typing import Any
from pathlib import Path
import data
import torch 
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as logging
from cosmos_tokenizer.networks import TokenizerConfigs
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from einops import rearrange
import cv2
import wandb
from configs import BaseConfig
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.utils import (
    numpy2tensor
)
import tqdm
import datetime
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from accelerate import Accelerator
import torch.nn.functional as F
import shutil
from accelerate.utils import set_seed

torch.manual_seed(0) 
np.random.seed(0)
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

config = BaseConfig()

tokenizer_config = TokenizerConfigs[config.tokenizer_type].value
tokenizer_config.update(dict(spatial_compression=config.spatial_compression))
if "I" in config.tokenizer_type:
    model_name = f"Cosmos-Tokenizer-{config.tokenizer_type}{config.spatial_compression}x{config.spatial_compression}"
else:
    model_name = f"Cosmos-Tokenizer-{config.tokenizer_type}{config.spatial_compression}x{config.spatial_compression}x{config.spatial_compression}"

def get_dataloader_kwargs():
    """Get kwargs for DataLoader in distributed setting"""
    return {
        'pin_memory': True,
        'num_workers': 4,  # Increased from 0
        'prefetch_factor': 2,  # Added prefetch
        'persistent_workers': True,  # Keep workers alive between epochs
    }

val_dataset = data.RawImageDataset(config.wm1xgpt_val_dir)
val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=True, **get_dataloader_kwargs()) #  pin_memory=True, num_workers=0, prefetch_factor=None

noise_scheduler = DDPMScheduler(num_train_timesteps=config.noise_steps)
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 64, 128, 128, 256, 256),  # (128, 128, 256, 256, 512, 512) # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

vae = ImageTokenizer(
        checkpoint=Path(config.pretrained_models) / model_name / "autoencoder.jit",
        checkpoint_enc=Path(config.pretrained_models) / model_name / "encoder.jit",
        checkpoint_dec=Path(config.pretrained_models) / model_name / "decoder.jit",
        tokenizer_config=tokenizer_config,
        device=None,
        dtype=config.dtype,
    )

def transform(batch, accelerator):
    input_imgs = batch
    input_imgs = rearrange(input_imgs, 'b h w c -> b c h w')
    input_imgs = input_imgs.to(accelerator.device).to(torch.bfloat16)
    input_imgs = input_imgs/_UINT8_MAX_F * 2.0 - 1.0
    return {"imgs": input_imgs}

def decode_img(preds):
    output_imgs = (preds.float() + 1.0) / 2.0
    output_imgs = rearrange(output_imgs, 'b c h w -> b h w c')
    output_imgs = output_imgs.clamp(0, 1).cpu().numpy()
    output_imgs = output_imgs * _UINT8_MAX_F + 0.5
    output_imgs = output_imgs.astype(np.uint8)
    return output_imgs

def sample_imgs(eval_dir, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=16,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)
    # Save the images
    test_dir = os.path.join(eval_dir)
    os.makedirs(test_dir, exist_ok=True)
    save_path = f"{test_dir}/sample_imgs.png"
    image_grid.save(save_path)


device = config.eval_device
accelerator = Accelerator()
model, val_dataloader = accelerator.prepare(
        model,  val_dataloader
    )
accelerator.load_state(config.resume_model)
# model.load_state_dict(torch.load(config.resume_model))
# model.eval()
# model.to(device)
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
path = Path(config.results_dir)
sample_imgs( path, pipeline)


# for i,batch in enumerate(val_dataloader):
#     transformed_batch = transform(batch)
#     input_imgs = transformed_batch["imgs"]

#     reconstructed = vae.autoencode(input_imgs)
#     output_imgs = decode_img(reconstructed)

#     plt.imsave('debug/debug_reconstructed.png', output_imgs[0])
#     psnrs = []
#     for i in range(input_imgs.shape[0]):
#         psnrs.append(cv2.PSNR(batch[i].numpy(),output_imgs[i]))
#     print('Upper bound PSNR', np.mean(psnrs).item(), np.std(psnrs).item())
#     break