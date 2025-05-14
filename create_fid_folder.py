import datetime
import os
from pathlib import Path

import cv2  # If sample_video returns numpy arrays or you want to save intermediates
import hydra
import numpy as np
import safetensors
import torch
import torchvision.transforms as transforms
import tqdm

# Re-introduce Accelerator related imports
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed  # Use accelerate's set_seed
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from einops import rearrange
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from PIL import Image  # For saving images
from torch.utils.data import DataLoader

# Assuming these modules are available in your environment
import data
import samplers
from conditioning import ConditioningManager
from data import (  # Assuming this can return test_loader directly
    encode_batch,
    get_dataloaders,
)
from models import get_model
from schedulers import get_scheduler

# --- Configuration ---
# Ensure these are in your Hydra config file (e.g., flow_video_mmdit.yaml)
#
# inference:
#   checkpoint_path: "/path/to/your/checkpoint_directory_or_file" # CRITICAL: Set correctly
#       # If use_ema=false, this should be the *directory* saved by accelerator.save_state
#       # If use_ema=true, this should be the *path* (file or dir) containing EMA weights saved by accelerator.save_model
#   use_ema: false # Set true to load EMA weights, false to load standard checkpoint state
#   batch_size: 4
#   output_dir: "/pub0/qasim/1xgpt/humanoid_world_model/submissions/diffusion/"
#   guidance_scale: 7.0
#   gen_type: "video"
#   num_frames_to_generate: 16

# Update data section in config to point to TEST data
# data:
#   type: 'humanoid_test'
#   hmwm_test_dir: "/path/to/test/data"
#   num_past_frames: 4
#   num_future_frames: 12
#   num_workers: 2 # Adjust as needed

# General config
# seed: 42
# image_size: 256
# mixed_precision: "no" # Or "fp16", "bf16" - Accelerator will handle this


def setup_accelerator(config):
    """Setup Accelerator for single-process inference."""
    # Initialize Accelerator. It will automatically detect 'NO' distributed type
    # if not launched via `accelerate launch` or if only one GPU is visible.
    # Specify mixed precision from config.
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),
        # No need to specify gradient_accumulation_steps for inference
        # No need for InitProcessGroupKwargs if running single process
    )
    # Set seed using accelerate's utility function for consistency across devices/processes
    set_seed(config.get("seed", 42))
    logging.info(f"Accelerator initialized on device: {accelerator.device}")
    logging.info(f"Number of processes: {accelerator.num_processes}")
    logging.info(f"Mixed precision: {accelerator.mixed_precision}")
    return accelerator


@hydra.main(
    config_path="configs", config_name="flow_video_mmdit_futureframe", version_base=None
)
def main(cfg: DictConfig):
    # --- Setup Accelerator and Seed ---
    accelerator = setup_accelerator(cfg)
    seed = cfg.get("seed", 42)  # Keep seed accessible if needed elsewhere

    logging.info("--- Starting Single-Process Inference Script (using Accelerator) ---")
    logging.info(f"Output directory: {cfg.inference.output_dir}")
    logging.info(f"Checkpoint path: {cfg.inference.checkpoint_path}")
    logging.info(f"Use EMA weights: {cfg.inference.use_ema}")
    logging.info(f"Inference batch size: {cfg.inference.batch_size}")
    logging.info(f"Guidance scale: {cfg.inference.guidance_scale}")

    # --- Load VAEs/Tokenizers ---
    logging.disable("cosmos_tokenizer")  # turnoff logging
    img_vae = None
    # Make sure dtype is compatible with accelerator.mixed_precision if set
    vae_dtype_str = cfg.image_tokenizer.get("dtype", "fp32")
    vae_dtype = torch.float32
    if vae_dtype_str == "fp16":
        vae_dtype = torch.float16
    elif vae_dtype_str == "bf16":
        vae_dtype = torch.bfloat16

    if cfg.image_tokenizer.get("path") and cfg.image_tokenizer.get(
        "use_img_vae", False
    ):
        model_name_img = f"Cosmos-Tokenizer-CI{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        img_vae = ImageTokenizer(
            checkpoint=Path(cfg.image_tokenizer.path)
            / model_name_img
            / "autoencoder.jit",
            checkpoint_enc=Path(cfg.image_tokenizer.path)
            / model_name_img
            / "encoder.jit",
            checkpoint_dec=Path(cfg.image_tokenizer.path)
            / model_name_img
            / "decoder.jit",
            device=None,  # Accelerator handles placement
            dtype=vae_dtype_str,  # Pass string representation or handle dtype object
        )
        img_vae.eval()
        for param in img_vae.parameters():
            param.requires_grad = False
        logging.info("Image VAE loaded.")

    model_name_vid = f"Cosmos-Tokenizer-CV{cfg.image_tokenizer.temporal_compression}x{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
    vid_vae = CausalVideoTokenizer(
        checkpoint=Path(cfg.image_tokenizer.path) / model_name_vid / "autoencoder.jit",
        checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name_vid / "encoder.jit",
        checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name_vid / "decoder.jit",
        device=None,  # Accelerator handles placement
        dtype="bfloat16",  # Keep consistent with training script example, ensure compatible
    )
    vid_vae.eval()
    for param in vid_vae.parameters():
        param.requires_grad = False
    logging.info("Video VAE loaded.")

    if img_vae is None:
        img_vae = vid_vae
        logging.info("Using Video VAE as Image VAE.")

    # --- Load Conditioning ---
    conditioning_manager = ConditioningManager(cfg.conditioning)
    # Don't move manually, let accelerator.prepare handle it
    logging.info("Conditioning manager loaded.")

    # --- Load Dataloader ---
    # Note: VAEs are passed here *before* being prepared by accelerator.
    # Ensure the dataloader logic doesn't require them to be on device already.
    logging.info("Loading test dataloader...")
    test_dataloader = get_dataloaders(
        "1xgpt_test",  # Assuming this is the correct type for your test data
        cfg,
        hmwm_train_dir=None,
        hmwm_val_dir=None,
        coco_train_imgs=None,
        coco_train_ann=None,
        coco_val_imgs=None,
        coco_val_ann=None,
        val_batch_size=None,
        vae=vid_vae,  # Pass original instances
        img_vae=img_vae,
        vid_vae=vid_vae,
        hmwm_test_dir=cfg.data.get("hmwm_test_dir"),
        image_size=cfg.image_size,
        train_batch_size=cfg.inference.batch_size,  # Pass BS for consistency
        test_batch_size=cfg.inference.batch_size,
        conditioning_type=cfg.conditioning.type,
        conditioning_manager=conditioning_manager,  # Pass original instance
        num_past_frames=cfg.conditioning.get("num_past_frames"),
        num_future_frames=cfg.inference.get(
            "num_frames_to_generate", cfg.conditioning.get("num_future_frames")
        ),
    )
    logging.info(f"Test dataloader loaded with {len(test_dataloader.dataset)} samples.")
    # --- Load Noise Scheduler ---
    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)
    logging.info(f"Noise scheduler '{cfg.model.scheduler_type}' loaded.")

    # --- Load Model ---
    latent_h = cfg.image_size // cfg.image_tokenizer.spatial_compression
    latent_w = cfg.image_size // cfg.image_tokenizer.spatial_compression
    try:
        latent_channels = vid_vae.config["latent_channels"]
    except (AttributeError, KeyError):
        latent_channels = cfg.model.get("latent_channels", 16)
        logging.warning(
            f"Could not infer latent_channels from VAE config, using value from model config: {latent_channels}"
        )

    logging.info(f"Inferred latent size: {latent_channels} x {latent_h} x {latent_w}")

    model = get_model(cfg, latent_channels, conditioning_manager, latent_h)
    logging.info(f"Model '{cfg.model.type}' instantiated.")

    # --- Prepare with Accelerator ---
    # This moves models/VAEs to the correct device and handles mixed precision
    logging.info(
        "Preparing model, VAEs, conditioning manager, and dataloader with Accelerator..."
    )
    model, vid_vae, img_vae, conditioning_manager, test_dataloader = (
        accelerator.prepare(
            model, vid_vae, img_vae, conditioning_manager, test_dataloader
        )
    )
    # Note: From here on, use the *prepared* objects.
    logging.info("Preparation complete.")

    # --- Load Checkpoint ---
    checkpoint_path = Path(cfg.inference.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    try:
        state_dict_loaded_manually = False
        if cfg.inference.use_ema:
            # Manual loading required for EMA, check file type
            checkpoint_path_str = str(checkpoint_path)
            if checkpoint_path_str.endswith(".safetensors"):
                # *** LOAD SAFETENSORS ***
                logging.info("Loading EMA weights from .safetensors file.")
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Expected .safetensors file at {checkpoint_path}"
                    )
                state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
                state_dict_loaded_manually = True
            elif checkpoint_path.is_file():  # Assume .bin, .pth, etc.
                # *** LOAD standard torch file ***
                logging.info(
                    "Loading EMA weights from standard PyTorch file (.bin, .pth, etc.)."
                )
                try:
                    # Use weights_only=None for PyTorch >= 2.6 compatibility (respects default)
                    state_dict = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=None
                    )
                except RuntimeError as e:
                    # Handle specific weights_only error ONLY IF source is trusted
                    if "weights_only" in str(e):
                        logging.warning(
                            f"torch.load with weights_only=None failed: {e}. Retrying with weights_only=False. ENSURE THIS FILE IS FROM A TRUSTED SOURCE."
                        )
                        state_dict = torch.load(
                            checkpoint_path, map_location="cpu", weights_only=False
                        )
                    else:
                        raise e  # Re-raise other torch.load errors
                state_dict_loaded_manually = True
            else:
                # Maybe the EMA path is a directory saved differently? Add handling if needed.
                raise ValueError(
                    f"EMA checkpoint path is not a recognized file type (.safetensors, .bin, .pth): {checkpoint_path}"
                )

            # If EMA state_dict was loaded manually, apply it to the unwrapped model
            if state_dict_loaded_manually:
                unwrapped_model = accelerator.unwrap_model(model)
                # Clean state dict (nesting/prefix) before loading
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                cleaned_state_dict = {}
                is_prefixed = all(k.startswith("module.") for k in state_dict.keys())
                if is_prefixed:
                    logging.info(
                        "Removing 'module.' prefix from manually loaded checkpoint keys."
                    )
                    for k, v in state_dict.items():
                        cleaned_state_dict[k[len("module.") :]] = v
                else:
                    cleaned_state_dict = state_dict

                missing_keys, unexpected_keys = unwrapped_model.load_state_dict(
                    cleaned_state_dict, strict=False
                )
                if missing_keys:
                    logging.warning(f"Manual Load: Missing keys: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Manual Load: Unexpected keys: {unexpected_keys}")
                logging.info("Manual Load: Weights loaded into unwrapped model.")

        else:
            # Load standard full checkpoint using accelerator.load_state
            if not checkpoint_path.is_dir():
                raise ValueError(
                    f"Expected a directory for accelerator.load_state (use_ema=False), but got path: {checkpoint_path}"
                )
            logging.info(
                f"Loading checkpoint state using accelerator.load_state from directory: {checkpoint_path}"
            )
            accelerator.load_state(checkpoint_path)
            logging.info("Checkpoint loaded via accelerator.load_state.")

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise
    # --- Set Model to Evaluation Mode ---
    model.eval()
    logging.info("Model set to evaluation mode.")

    # --- Instantiate Sampler ---
    # Pass the *unwrapped* VAE after it has been prepared (moved to device)
    vae_unwrapped = accelerator.unwrap_model(vid_vae)
    sampler = samplers.Sampler(
        vae=vae_unwrapped,
        scheduler=noise_scheduler,
        seed=seed,
        spatial_compression=cfg.image_tokenizer.spatial_compression,
        num_inference_steps=cfg.model.noise_steps,
        latent_channels=latent_channels,
    )
    logging.info("Sampler instantiated.")

    # --- Prepare Output Directory ---
    output_dir = Path(cfg.inference.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")

    # --- Inference Loop ---
    if accelerator.is_main_process:
        progress_bar = tqdm.tqdm(
            total=len(test_dataloader),
            desc="Generating Samples",
            disable=not accelerator.is_local_main_process,
        )

    for step, batch in enumerate(test_dataloader):
        # Batches from the prepared dataloader are automatically on the correct device
        with torch.no_grad():
            sampling_dtype = (
                torch.float32
            )  # Or adjust based on model/mixed precision needs

            # --- Select Sampler ---
            if "future_frame" in cfg.gen_type.lower():
                logging.debug(f"Calling sample_video for batch {step}")
                # Pass the *unwrapped* model to the sampler
                model_unwrapped = accelerator.unwrap_model(model)
                # VAE passed during sampler init was already unwrapped
                if len(batch["future_frames"].shape) == 4:
                    batch["future_frames"] = batch["future_frames"].unsqueeze(
                        2
                    )  # Ensure dtype is correct
                # batch['past_actions'] = batch['past_actions'].unsqueeze(0) # Ensure dtype is correct
                # Run sampling
                sample_videos = sampler.sample_future_frame(  # Ignoring sample_grids
                    cfg=cfg,
                    dataloader=test_dataloader,  # Pass prepared dataloader maybe? Check sampler needs
                    batch=batch,  # Pass batch (already on device)
                    vid_vae=vae_unwrapped,
                    img_vae=img_vae,
                    accelerator=accelerator,  # Pass accelerator instance if needed by sampler
                    model=model_unwrapped,  # Pass unwrapped model
                    guidance_scale=cfg.inference.guidance_scale,
                    n_samples=cfg.inference.batch_size,
                    dtype=sampling_dtype,
                    device=accelerator.device,  # Use accelerator's device
                )
                # sample_videos: list (batch size) of lists (frames) of PIL Images
            elif "video" in cfg.gen_type.lower():
                logging.debug(f"Calling sample_video for batch {step}")
                # Pass the *unwrapped* model to the sampler
                model_unwrapped = accelerator.unwrap_model(model)
                sample_videos, _ = sampler.sample_video(  # Ignoring sample_grids
                    cfg=cfg,
                    dataloader=test_dataloader,  # Pass prepared dataloader maybe? Check sampler needs
                    batch=batch,  # Pass batch (already on device)
                    vid_vae=vae_unwrapped,
                    img_vae=img_vae,
                    accelerator=accelerator,  # Pass accelerator instance if needed by sampler
                    model=model_unwrapped,  # Pass unwrapped model
                    guidance_scale=cfg.inference.guidance_scale,
                    n_samples=cfg.inference.batch_size,
                    dtype=sampling_dtype,
                    device=accelerator.device,  # Use accelerator's device
                )
                # sample_videos: list (batch size) of lists (frames) of PIL Images
            else:
                if accelerator.is_main_process:
                    logging.warning(
                        f"Unsupported generation type '{cfg.gen_type}'. Skipping batch {step}."
                    )
                if accelerator.is_main_process:
                    progress_bar.update(1)
                continue

            # --- Process and Save Results ---
            # Only the main process saves the files
            if accelerator.is_main_process:
                if "sample_id" not in batch:
                    logging.error(
                        f"Batch {step} does not contain 'sample_id'. Cannot save files correctly."
                    )
                    progress_bar.update(1)
                    continue

                batch_sample_ids = batch["sample_id"]
                if isinstance(batch_sample_ids, torch.Tensor):
                    # Ensure IDs are on CPU for list conversion/file naming
                    batch_sample_ids = batch_sample_ids.cpu().tolist()

                if len(sample_videos) != len(batch_sample_ids):
                    logging.warning(
                        f"Mismatch in batch {step}: Got {len(sample_videos)} videos but {len(batch_sample_ids)} sample IDs. Skipping save."
                    )
                    progress_bar.update(1)
                    continue

                for i in range(len(batch_sample_ids)):
                    sample_id = batch_sample_ids[i]
                    individual_video_frames = sample_videos[i]

                    if not individual_video_frames:
                        logging.warning(
                            f"No frames generated for sample_id {sample_id}. Skipping."
                        )
                        continue

                    frame_to_save = individual_video_frames  # Last frame
                    output_filename = output_dir / f"{sample_id}.png"
                    try:
                        frame_to_save.save(output_filename, format="PNG")
                        logging.debug(
                            f"Saved image for sample {sample_id} to {output_filename}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Failed to save image for sample {sample_id} to {output_filename}: {e}"
                        )

        # Update progress bar on the main process
        if accelerator.is_main_process:
            progress_bar.update(1)

    if accelerator.is_main_process:
        progress_bar.close()
        logging.info("--- Inference Complete ---")


if __name__ == "__main__":
    main()
