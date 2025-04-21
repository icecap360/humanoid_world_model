import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import hydra
import datetime
import shutil
import cv2
import torch.nn.functional as F

# Remove unused imports from training (e.g. wandb, optimizer, backprop)
from loguru import logger
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from data import get_dataloaders, encode_batch
from conditioning import ConditioningManager
from models import get_model
import samplers
from diffusers.utils import make_image_grid
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from schedulers import get_scheduler

def compute_val_loss(cfg, model, val_dataloader, vae_model, noise_scheduler, accelerator, progress_bar_enabled=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    n_timesteps = cfg.model.noise_steps

    if accelerator.is_main_process and progress_bar_enabled:
        pbar = tqdm.tqdm(total=len(val_dataloader), desc="Validating")
    else:
        pbar = None

    for step, batch in enumerate(val_dataloader):
        # Encode the batch into latents
        latents, batch = encode_batch(cfg, batch, vae_model, accelerator)
        bs = latents.shape[0]
        noise = torch.randn(latents.shape, device=accelerator.device)
        timesteps = torch.randint(
            0, n_timesteps, (bs, 1), device=accelerator.device, dtype=torch.int64
        )
        # Add noise to the latents using the noise scheduler
        batch['noisy_latents'] = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            noise_pred = model(batch, timesteps, accelerator.device, use_cfg=True)
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
        total_loss += loss.item() * bs
        total_samples += bs

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

    # Gather metrics across processes
    total_loss_tensor, total_samples_tensor = accelerator.gather_for_metrics(
        (torch.tensor(total_loss, device=accelerator.device),
         torch.tensor(total_samples, device=accelerator.device))
    )
    avg_loss = total_loss_tensor.sum() / total_samples_tensor.sum()
    return avg_loss.item()

@hydra.main(config_path="configs", config_name="flow_video_mmdit", version_base=None)
def main(cfg):
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    set_seed(cfg.seed)
    logger.info(f'Configuration: {cfg}')

    # Setup accelerator (for distributed evaluation if needed)
    accelerator = Accelerator()

    # Setup the VAE from cosmos_tokenizer
    tokenizer_config = TokenizerConfigs[cfg.image_tokenizer.tokenizer_type].value
    tokenizer_config.update(dict(spatial_compression=cfg.image_tokenizer.spatial_compression))
    logger.disable('cosmos_tokenizer')

    if "I" in cfg.image_tokenizer.tokenizer_type:
        model_name = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = ImageTokenizer(
            checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
            checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
            checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
            tokenizer_config=tokenizer_config,
            device=None,
            dtype=cfg.image_tokenizer.dtype,
        )
    else:
        model_name = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.temporal_compression}x{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = CausalVideoTokenizer(
            checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
            checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
            checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
            tokenizer_config=tokenizer_config,
            device=None,
            dtype="bfloat16",
        )

    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False

    conditioning_manager = ConditioningManager(cfg.conditioning)

    # Get dataloaders (we only need the validation dataloader)
    train_dataloader, val_dataloader = get_dataloaders(
        cfg.data.type,
        cfg,
        vae=vae.module if hasattr(vae, "module") else vae,
        hmwm_train_dir=cfg.data.hmwm_train_dir,
        hmwm_val_dir=cfg.data.hmwm_val_dir,
        coco_train_imgs=cfg.data.coco_train_imgs,
        coco_val_imgs=cfg.data.coco_val_imgs,
        coco_train_ann=cfg.data.coco_train_ann,
        coco_val_ann=cfg.data.coco_val_ann,
        image_size=cfg.image_size,
        train_batch_size=cfg.train.batch_size,
        val_batch_size=cfg.val.batch_size,
        conditioning_type=cfg.conditioning.type,
        conditioning_manager=conditioning_manager,
        num_past_frames=cfg.conditioning.get('num_past_frames') + cfg.conditioning.get('num_future_frames'),
        num_future_frames=cfg.conditioning.get('num_future_frames'),
    )

    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)
    latent_channels = tokenizer_config["latent_channels"]

    # Load the model
    model = get_model(cfg, latent_channels, conditioning_manager, cfg.image_size // cfg.image_tokenizer.spatial_compression)

    # Prepare model, vae and dataloader for accelerator
    model, vae, val_dataloader = accelerator.prepare(model, vae, val_dataloader)
    conditioning_manager.to(accelerator.device)

    accelerator.load_state(cfg.train.resume_model)

    # Compute and print the validation loss
    vae_model = vae.module if hasattr(vae, "module") else vae
    # val_loss = compute_val_loss(cfg, model, val_dataloader, vae_model, noise_scheduler, accelerator)
    # if accelerator.is_main_process:
    #     print(f"Validation loss: {val_loss:.6f}")

    # Save image samples to directory "eval_diffusion"
    sampler = samplers.Sampler(vae, noise_scheduler, cfg.seed, cfg.image_tokenizer.spatial_compression, latent_channels, cfg.model.noise_steps)
    eval_dir = Path("eval_diffusion")
    os.makedirs(eval_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Branch based on generation type; adjust as needed.
        if 'video' in cfg.gen_type.lower():
            sample_videos, sample_grids = sampler.sample_video_autoregressive(
                cfg,
                train_dataloader,
                batch_idx=0,
                vae=vae_model,
                accelerator=accelerator,
                model=model,
                guidance_scale=cfg.model.cfg_scale,
                dtype=torch.float32,
                device=accelerator.device
            )
            os.makedirs(eval_dir, exist_ok=True)
            b = len(sample_videos)
            h, w = sample_videos[0][0].size
            for i in range(b):
                path_vid = os.path.join(eval_dir, f'output-{i}.mp4')
                out = cv2.VideoWriter(path_vid, cv2.VideoWriter_fourcc(*'mp4v'), 0.05, (w, h))
                for frame in sample_videos[i]:
                    out.write(np.asarray(frame))
                out.release()
                path_img0 = os.path.join(eval_dir, f'tokenizedprompt+predictions.jpeg')
                sample_grids[0][i].save(path_img0)
                
                path_img1 = os.path.join(eval_dir, f'gtprompt+predictions.jpeg')
                sample_grids[1][i].save(path_img1)
                
                path_img2 = os.path.join(eval_dir, f'gt_vs_predictions.jpeg')
                sample_grids[2][i].save(path_img2)
            print(f"Saved video sample to: {path_vid}")
        elif cfg.conditioning.type == 'text':
            samples = sampler.sample_textcond_img(
                model=model,
                img_size=cfg.image_size,
                in_channels=latent_channels,
                prompt_file=cfg.conditioning.prompt_file,
                text_tokenizer=conditioning_manager.get_module()['text'],
                device=accelerator.device,
                dtype=torch.float32,
                guidance_scale=cfg.model.cfg_scale
            )
            image_grid = make_image_grid(samples, rows=4, cols=4)
            samples_path = os.path.join(eval_dir, 'samples.png')
            image_grid.save(samples_path)
            print(f"Saved image samples to: {samples_path}")
        else:
            samples = sampler.sample_uncond_img(
                unet=model,
                img_size=cfg.image_size,
                in_channels=latent_channels,
                device=accelerator.device,
                dtype=torch.float32
            )
            image_grid = make_image_grid(samples, rows=4, cols=4)
            samples_path = os.path.join(eval_dir, 'samples.png')
            image_grid.save(samples_path)
            print(f"Saved image samples to: {samples_path}")

if __name__ == '__main__':
    main()
