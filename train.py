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
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.utils import (
    numpy2tensor
)
import hydra
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
import tqdm
import datetime
from diffusers.optimization import get_cosine_schedule_with_warmup
from schedulers import get_scheduler
from accelerate import Accelerator, InitProcessGroupKwargs
import torch.nn.functional as F
import shutil
from accelerate.utils import set_seed
import samplers
from diffusers.utils import make_image_grid
from loguru import logger
from models.unet import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from conditioning import ConditioningManager
from data import get_dataloaders, encode_batch
from models import get_model
from diffusers.training_utils import EMAModel
from accelerate.utils import DistributedDataParallelKwargs

def setup_distributed_training(config):
    """Setup for distributed training"""
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_dir=config.log_dir,
        # Add these parameters for multi-GPU
        split_batches=False,  # Split batches across devices
        device_placement=True,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=18000))]
    )
    # Set seed for reproducibility
    set_seed(config.seed)
    return accelerator

def compute_val_loss(model, val_dataloader, vae_model, noise_scheduler, accelerator, n_timesteps, cfg,  progress_bar_enabled=True):
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
        # if isinstance(batch, (list, tuple)):
        #     if len(batch) == 2:
        #         imgs, captions = batch[0], batch[1]
        #     else:
        #         imgs = batch
        #         captions = None
        # elif isinstance(batch, dict):
        #     batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        #     imgs = batch["image"]
        #     captions = batch.get("caption", None)
        # else:
        #     imgs = batch
        #     captions = None

        latents, batch = encode_batch(cfg, batch, vae_model, accelerator)
        # imgs = batch["imgs"]
        # # Move images to the accelerator device and convert to bfloat16
        # imgs = imgs.to(accelerator.device).to(torch.bfloat16)
        # # Encode images to latents
        # latents = vae_model.encode(imgs)[0]

        bs = latents.shape[0]
        # Generate noise and timesteps
        noise = torch.randn(latents.shape, device=accelerator.device)
        timesteps = torch.randint(
            0,
            n_timesteps,
            (bs,1),
            device=accelerator.device,
            dtype=torch.int64
        )
        # Add noise to latents
        batch['noisy_latents'] = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            # Predict the noise residual
            noise_pred = model(batch, timesteps, accelerator.device, use_cfg=True)
            # Compute mean squared error loss (averaged over the batch)
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        total_loss += loss.item() * bs
        total_samples += bs

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
        break
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

@hydra.main(config_path="configs", config_name="flow_video_mmdit", version_base=None)
def main(cfg):
    torch.manual_seed(0) 
    np.random.seed(0)

    logging.info(f'debug: {cfg.debug}')
    logging.info(f'learning_rate: {cfg.train.learning_rate}')
    logging.info(f'lr_warmup_steps: {cfg.train.lr_warmup_steps}')    
    logging.info(f'one_sample: {cfg.one_sample}')
    logging.info(f'image_size: {cfg.image_size}')
    logging.info(f'log_dir: {cfg.log_dir}')
    logging.info(f'train_batch_size: {cfg.train.batch_size}')
    logging.info(f'val_iters: {cfg.train.val_iters}')
    logging.info(f'exp_prefix: {cfg.exp_prefix}')
    logging.info(f'data: {cfg.data.type}')
    logging.info(f'conditioning: {cfg.conditioning.type}')
    logging.info(f'model: {cfg.model.type}')

    if cfg.one_sample:
        cfg.train.learning_rate = 8e-5
        cfg.train.lr_warmup_steps = 0
        cfg.train.gradient_accumulation_steps = 1
        cfg.train.batch_size = 4
        cfg.val.run = True
        cfg.exp_prefix = 'one-sample'
        cfg.train.save_model_iters = 6000
        cfg.train.val_iters = 5
        cfg.conditioning.prompt_file = 'prompts_one_sample.txt'
        cfg.val.skip_val_loss = False

    tokenizer_config = TokenizerConfigs[cfg.image_tokenizer.tokenizer_type].value
    tokenizer_config.update(dict(spatial_compression=cfg.image_tokenizer.spatial_compression))
    logger.disable('cosmos_tokenizer') # turnoff logging
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

    for param in vae.parameters():
        param.requires_grad = False

    conditioning_manager = ConditioningManager(cfg.conditioning)

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
        num_past_frames=cfg.conditioning.get('num_past_frames'),
        num_future_frames=cfg.conditioning.get('num_future_frames'),
    )
    # first_batch = next(iter(train_dataloader))

    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)

    latent_channels= tokenizer_config["latent_channels"] # 3

    model = get_model(cfg, latent_channels, conditioning_manager, cfg.image_size // cfg.image_tokenizer.spatial_compression)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.train.num_epochs),
    )
    
    sampler = samplers.Sampler(vae, noise_scheduler, cfg.seed, cfg.image_tokenizer.spatial_compression, latent_channels, cfg.model.noise_steps)

    accelerator = setup_distributed_training(cfg)
    if accelerator.is_main_process and not cfg.debug:
        wandb.init(
            project="diffusion-video-generator",
            name=f"{cfg.exp_prefix}: {datetime.datetime.now().strftime('%B-%d %H-%M-%S')}",
            config={
                "learning_rate": cfg.train.learning_rate,
                "num_epochs": cfg.train.num_epochs,
                "train_batch_size": cfg.train.batch_size,
                "eval_batch_size": cfg.val.batch_size,
                "noise_steps": cfg.model.noise_steps,
                "image_size": cfg.image_size,
                "mixed_precision": cfg.mixed_precision,
            }
        )
        if cfg.log_dir is not None and not cfg.debug:
            os.makedirs(cfg.log_dir, exist_ok=True)
        accelerator.init_trackers("train_ddpm")

    model, vae, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, vae, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    conditioning_manager.to(accelerator.device)
    
    # Compile the model
    if torch.cuda.is_available() and not cfg.debug:
        model = torch.compile(model) #defaults to mode="reduce-overhead"

    ema_model = EMAModel(
        model,
        inv_gamma=cfg.model.ema_inv_gamma,
        power=cfg.model.ema_power,
        max_value=cfg.model.ema_max_decay
    )

    start_epoch, start_iter = 0, 0
    if cfg.train.resume_train:
        accelerator.load_state(cfg.train.resume_model)
        path = os.path.basename(cfg.train.resume_model)
        start_epoch, start_iter = int(path.split('-')[1]), int(path.split('-')[2]) 
    
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%B-%d-%H-%M")
    eval_dir = Path(cfg.log_dir) / (cfg.model.scheduler_type + ' ' + formatted_datetime)
    shutil.rmtree(cfg.log_dir, ignore_errors=True)

    vae_model = vae.module if hasattr(vae, "module") else vae
    if cfg.one_sample:
        first_batch = next(iter(train_dataloader))
    global_step = 0
    grad_norm = 0.0

    for epoch in range(start_epoch, cfg.train.num_epochs):
        if accelerator.is_main_process:
            progress_bar = tqdm.tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if step < start_iter:
                continue

            if cfg.one_sample:
                batch = first_batch
            
            latents, batch = encode_batch(cfg, batch, vae_model, accelerator)
            noise = torch.randn(latents.shape, device=latents.device)
            bs = latents.shape[0]
            timesteps = torch.randint(
                    0, 
                    cfg.model.noise_steps, 
                    (bs,1), 
                    device=latents.device,
                    dtype=torch.int64
                )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            batch['noisy_latents'] = noisy_latents
            model.train()
            with accelerator.accumulate(model):
                # Predict the noise residual
                # noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]
                noise_pred = model(batch, timesteps, accelerator.device, use_cfg=True)
                noise_target = noise_scheduler.get_target(latents, noise, timesteps)
                loss = F.mse_loss(noise_pred, noise_target)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    grad_norm = grad_norm.detach().item()
                
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    ema_model.step(model)
                optimizer.zero_grad()
                
            if accelerator.is_main_process:
                logs = OrderedDict({
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm,
                    "train/epoch": epoch,
                    "train/step": global_step,
                })
                progress_bar.update(1)
                progress_bar.set_postfix(ordered_dict=logs)
                if not cfg.debug: 
                    wandb.log(logs)
                if step % cfg.train.save_model_iters == 0 and step > 0 and not cfg.debug:
                    os.makedirs(eval_dir, exist_ok=True)    
                    path = eval_dir / ('checkpoint-' +str(epoch))
                    accelerator.save_state(path, safe_serialization=True)
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    accelerator.save_model(model,eval_dir / ('checkpoint-' +str(epoch)) / "ema")
                    ema_model.restore(model.parameters())

            if (step % cfg.train.val_iters == 0 or step == len(train_dataloader) - 1) and step > 0 and cfg.val.run:
                path = eval_dir / ('samples-' + str(epoch) + '-' + str(step) + '.png')
                if cfg.debug:
                    path = 'debug/debug.png'
                model_unwrapped = model # model # accelerator.unwrap_model(model)
                model_unwrapped.eval()

                # Log validation loss
                if not cfg.val.skip_val_loss:
                    val_loss = compute_val_loss(model_unwrapped, val_dataloader, vae_model, noise_scheduler,  accelerator, cfg.model.noise_steps, cfg)
                    if accelerator.is_main_process:
                        if not cfg.debug:
                            wandb.log({
                                "validation/loss": val_loss,
                                "validation/epoch": epoch,
                                "validation/step": step
                            })
                        logging.info(f"Validation loss: {val_loss:.6f}")
                
                # Sample some images
                if accelerator.is_main_process and not cfg.val.skip_img_sample:
                    with torch.no_grad():
                        if cfg.conditioning.type == 'text':
                            samples = sampler.sample_text(
                                model=model_unwrapped, 
                                img_size=cfg.image_size, 
                                in_channels=latent_channels,
                                prompt_file=cfg.conditioning.prompt_file,
                                text_tokenizer=conditioning_manager.get_module()['text'],
                                device=accelerator.device, 
                                dtype=noisy_latents.dtype,
                                guidance_scale=cfg.model.cfg_scale)
                        else:
                            samples = sampler.sample(
                                unet=model_unwrapped, 
                                img_size=cfg.image_size, 
                                in_channels=latent_channels,
                                device=accelerator.device, 
                                dtype=noisy_latents.dtype)
                    
                    image_grid = make_image_grid(samples, rows=4, cols=4)
                    os.makedirs(eval_dir, exist_ok=True)
                    image_grid.save(path)
                    logging.info(f"Sampled images to : {path}")
                    if accelerator.is_main_process and not cfg.debug:
                        wandb.log({
                            "generated_images": wandb.Image(str(path), caption=f"Epoch {epoch}, iter {step}")
                        })
            global_step += 1

        start_iter = 0
        if accelerator.is_main_process and not cfg.debug:
            os.makedirs(eval_dir, exist_ok=True)    
            path = eval_dir / ('checkpoint-' +str(epoch))
            accelerator.save_state(path, safe_serialization=True)
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
            accelerator.save_model(model, eval_dir / ('checkpoint-' +str(epoch)) / "ema")
            ema_model.restore(model.parameters())

    if accelerator.is_main_process and not cfg.debug:
        wandb.finish()


if __name__ == '__main__':
    main()