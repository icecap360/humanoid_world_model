import os
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
from accelerate import Accelerator, InitProcessGroupKwargs
import torch.nn.functional as F
import shutil
from accelerate.utils import set_seed
import samplers
from diffusers.utils import make_image_grid
import argparse
from models.unet import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from models.tokenizers import T5TextEmbedder, CLIPTextEmbedder

def parse_arguments():
    parser = argparse.ArgumentParser(description="My training script.")

    # Add the debug argument
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction, # Preferred for Python 3.9+
        default=False,
        help="Enable or disable Weights & Biases logging (default: True)",
    )

    # Example of other arguments you might have
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )

    return parser.parse_args()

torch.manual_seed(0) 
np.random.seed(0)
args = parse_arguments()

config = BaseConfig()
text_tokenizer = None

tokenizer_config = TokenizerConfigs[config.tokenizer_type].value
tokenizer_config.update(dict(spatial_compression=config.spatial_compression))
if "I" in config.tokenizer_type:
    model_name = f"Cosmos-Tokenizer-{config.tokenizer_type}{config.spatial_compression}x{config.spatial_compression}"
else:
    model_name = f"Cosmos-Tokenizer-{config.tokenizer_type}{config.spatial_compression}x{config.spatial_compression}x{config.spatial_compression}"

vae = ImageTokenizer(
        checkpoint=Path(config.pretrained_models) / model_name / "autoencoder.jit",
        checkpoint_enc=Path(config.pretrained_models) / model_name / "encoder.jit",
        checkpoint_dec=Path(config.pretrained_models) / model_name / "decoder.jit",
        tokenizer_config=tokenizer_config,
        device=None,
        dtype=config.dtype,
    )
for param in vae.parameters():
    param.requires_grad = False

def get_dataloader_kwargs():
    """Get kwargs for DataLoader in distributed setting"""
    return {
        'pin_memory': True,
        'num_workers': 4,  # Increased from 0
        'prefetch_factor': 2,  # Added prefetch
        'persistent_workers': True,  # Keep workers alive between epochs
    }

if config.data["type"] == "1xgpt":
    assert config.conditioning != 'text'
    train_dataset = data.RawImageDataset(config.data["1xgpt_train_dir"])
    val_dataset = data.RawImageDataset(config.data["1xgpt_val_dir"])
elif config.data["type"].lower() == "coco":
    assert config.conditioning == 'text'
    transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),  # Resize to fit model input (optional)
            transforms.ToTensor(),           # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    if config.text_tokenizer == 'T5':
        text_tokenizer = T5TextEmbedder()
    else:
        text_tokenizer = CLIPTextEmbedder()
    train_dataset = data.CustomCoco(
        root=config.data["coco_train_imgs"], 
        annFile=config.data["coco_train_ann"], 
        text_tokenizer=text_tokenizer,
        transform=transform)
    val_dataset = data.CustomCoco(
        root=config.data["coco_val_imgs"], 
        annFile=config.data["coco_val_ann"], 
        text_tokenizer=text_tokenizer,
        transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, **get_dataloader_kwargs()) #  pin_memory=True, num_workers=0, prefetch_factor=None
val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, **get_dataloader_kwargs()) #  pin_memory=True, num_workers=0, prefetch_factor=None

noise_scheduler = DDPMScheduler(num_train_timesteps=config.noise_steps)
latent_channels= tokenizer_config["latent_channels"] # 3
# model = UNet2DModel(
#     sample_size=config.image_size,  # the target image resolution
#     in_channels=latent_channels,  # the number of input channels, 3 for RGB images
#     out_channels=latent_channels,  # the number of output channels
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     block_out_channels= (224, 224*2, 224*3, 224*4), # (64, 64, 128, 128, 256, 256), # (128, 128, 256, 256, 512, 512), # the number of output channels for each UNet block
#     down_block_types=(
#         "DownBlock2D",  # a regular ResNet downsampling block
#         "DownBlock2D",
#         "DownBlock2D",
#         "AttnDownBlock2D",
#         # "DownBlock2D",
#         # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#         # "DownBlock2D",
#     ),
#     up_block_types=(
#         "UpBlock2D",  # a regular ResNet upsampling block
#         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#         "UpBlock2D",
#         "UpBlock2D",
#         # "UpBlock2D",
#         # "UpBlock2D",
#     ),
# )
model = UNet(latent_channels, 
             latent_channels, 
             config.unet_blocks, 
             128, 
             text_tokenizer=text_tokenizer,
             cfg_prob=config.cfg_prob,
             attention_resolutions=config.attention_resolutions)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
sampler = samplers.Sampler(vae, noise_scheduler, config.seed, config.spatial_compression, latent_channels, config.noise_steps)

def setup_distributed_training(config):
    """Setup for distributed training"""
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=config.log_dir,
        # Add these parameters for multi-GPU
        split_batches=False,  # Split batches across devices
        device_placement=True,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=18000))]
    )
    # Set seed for reproducibility
    set_seed(config.seed)
    return accelerator
accelerator = setup_distributed_training(config)
if accelerator.is_main_process and not args.debug:
    wandb.init(
        project="diffusion-video-generator",
        name=f"{config.exp_prefix}: {datetime.datetime.now().strftime('%B-%d %H-%M-%S')}",
        config={
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "noise_steps": config.noise_steps,
            "image_size": config.image_size,
            "mixed_precision": config.mixed_precision,
        }
    )
    if config.log_dir is not None and not (args.debug):
        os.makedirs(config.log_dir, exist_ok=True)
    accelerator.init_trackers("train_ddpm")

model, vae, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, vae, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

start_epoch, start_iter = 0, 0
if config.resume_train:
    accelerator.load_state(config.resume_model)
    path = os.path.basename(config.resume_model)
    start_epoch, start_iter = int(path.split('-')[1]), int(path.split('-')[2]) 
now = datetime.datetime.now()
formatted_datetime = now.strftime("%B-%d-%H-%M")
eval_dir = Path(config.eval_dir) / (config.exp_prefix + ' ' + formatted_datetime)
shutil.rmtree(eval_dir, ignore_errors=True)

vae_model = vae.module if hasattr(vae, "module") else vae

def validate(model, val_dataloader, vae_model, noise_scheduler, accelerator, progress_bar_enabled=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # Create progress bar only on the main process and if enabled
    if accelerator.is_main_process and progress_bar_enabled:
        pbar = tqdm.tqdm(total=len(val_dataloader), desc="Validating")
    else:
        pbar = None

    for step, batch in enumerate(val_dataloader):
        # Process batch based on its type
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                imgs, captions = batch[0], batch[1]
            else:
                imgs = batch
                captions = None
        elif isinstance(batch, dict):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            imgs = batch["image"]
            captions = batch.get("caption", None)
        else:
            imgs = batch
            captions = None

        # Move images to the accelerator device and convert to bfloat16
        imgs = imgs.to(accelerator.device).to(torch.bfloat16)
        # Encode images to latents
        latents = vae_model.encode(imgs)[0]
        bs = latents.shape[0]
        # Generate noise and timesteps
        noise = torch.randn(latents.shape, device=accelerator.device)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=accelerator.device,
            dtype=torch.int64
        )
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            # Predict the noise residual
            noise_pred = model(noisy_latents, timesteps, captions, use_cfg=True)
            # Compute mean squared error loss (averaged over the batch)
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        total_loss += loss.item() * bs
        total_samples += bs

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
    
    # if pbar is not None:
    #     pbar.close()

    # Gather metrics from all processes
    total_loss_tensor, total_samples_tensor = accelerator.gather_for_metrics(
        (torch.tensor(total_loss, device=accelerator.device), 
         torch.tensor(total_samples, device=accelerator.device))
    )
    # Compute average loss across all processes
    avg_loss = total_loss_tensor.sum() / total_samples_tensor.sum()
    return avg_loss.item()


for epoch in range(start_epoch, config.num_epochs):
    if accelerator.is_main_process:
        progress_bar = tqdm.tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dataloader):
        if step < start_iter:
            continue
        if isinstance(batch, list) and len(batch) == 2:
            imgs, captions = batch[0], batch[1]
        else:
            imgs = batch
            captions = None  # Handle case where meta_data is missing

        with accelerator.autocast():
            imgs = imgs.to(torch.bfloat16)
            latents = vae_model.encode(imgs)[0]
        global_step = 0

        noise = torch.randn(latents.shape, device=latents.device)
        bs = latents.shape[0]
        timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (bs,), 
                device=latents.device,
                dtype=torch.int64
            )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        model.train()
        with accelerator.accumulate(model):
            # Predict the noise residual
            # noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]
            noise_pred = model(noisy_latents, timesteps, captions, use_cfg=True)
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        logs = {
            "train/loss": loss.detach().item(),
            "train/lr": lr_scheduler.get_last_lr()[0],
            "train/epoch": epoch,
            "train/step": global_step,
        }
        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            if not args.debug: 
                wandb.log(logs)
        global_step += 1

        if step % config.eval_iters == 0 and step > 0:
            
            path = eval_dir / ('samples-' + str(epoch) + '-' + str(step) + '.png')
            if args.debug:
                path = 'debug/debug.png'
            model_unwrapped = model # accelerator.unwrap_model(model)
            model_unwrapped.eval()

            # Log validation loss
            val_loss = validate(model_unwrapped, val_dataloader, vae_model, noise_scheduler, accelerator)
            if accelerator.is_main_process:
                if not args.debug:
                    wandb.log({
                        "validation/loss": val_loss,
                        "validation/epoch": epoch,
                        "validation/step": step
                    })
                print(f"Validation loss: {val_loss:.6f}")
            
            # Sample some images
            if accelerator.is_main_process:
                with torch.no_grad():
                    if config.conditioning == 'text':
                        samples = sampler.sample_text(
                            model=model_unwrapped, 
                            img_size=config.image_size, 
                            in_channels=latent_channels, 
                            prompt_file=config.prompt_file,
                            text_tokenizer=text_tokenizer,
                            device=accelerator.device, 
                            dtype=noisy_latents.dtype)
                    else:
                        samples = sampler.sample(
                            unet=model_unwrapped, 
                            img_size=config.image_size, 
                            in_channels=latent_channels, 
                            device=accelerator.device, 
                            dtype=noisy_latents.dtype)
                image_grid = make_image_grid(samples, rows=4, cols=4)
                os.makedirs(eval_dir, exist_ok=True)
                image_grid.save(path)
                if accelerator.is_main_process and not args.debug:
                    wandb.log({
                        "generated_images": wandb.Image(str(path), caption=f"Epoch {epoch}, iter {step}")
                    })

    if accelerator.is_main_process and not args.debug:
        os.makedirs(eval_dir, exist_ok=True)    
        path = eval_dir / ('checkpoint-' +str(epoch))
        accelerator.save_state(path, safe_serialization=True)
    start_iter = 0



if accelerator.is_main_process and not args.debug:
    wandb.finish()

# for i,batch in enumerate(train_dataloader):
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
