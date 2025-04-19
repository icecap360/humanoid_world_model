import os
import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed, extract_model_from_parallel
import tqdm
import datetime
import shutil
from loguru import logger
import tempfile
from PIL import Image
import math

# Import necessary components from your training script's modules
import data # Assuming data.py contains get_dataloaders and encode_batch
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from models import get_model
from conditioning import ConditioningManager
from schedulers import get_scheduler
import samplers # Assuming samplers.py contains Sampler class
# No longer need EMAModel

# Import safetensors loader
from safetensors.torch import load_file

# Import clean-fid
from cleanfid import fid

# --- Configuration ---
@hydra.main(config_path="configs", config_name="flow_video_mmdit", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    gen_batch_size = cfg.val.batch_size # Batch size for generating videos
    fid_batch_size = 32 # Batch size for FID feature extraction (adjust based on GPU memory)
    
    # --- Setup Accelerator ---
    accelerator = Accelerator(device_placement=True)
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    # --- Load Config and Components (similar to training script) ---
    logger.info("Loading components...")
    
    # VAE (Assuming VAE loading remains the same)
    tokenizer_config = TokenizerConfigs[cfg.image_tokenizer.tokenizer_type].value
    tokenizer_config.update(dict(spatial_compression=cfg.image_tokenizer.spatial_compression))
    logger.disable('cosmos_tokenizer') 
    if "I" in cfg.image_tokenizer.tokenizer_type:
        model_name = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = ImageTokenizer( # Use appropriate class
             checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
             checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
             checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
             tokenizer_config=tokenizer_config, device=None, dtype=cfg.image_tokenizer.dtype)
    else:
        model_name = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.temporal_compression}x{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = CausalVideoTokenizer( # Use appropriate class
             checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
             checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
             checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
             tokenizer_config=tokenizer_config, device=None, dtype="bfloat16") # Or your preferred eval dtype
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Conditioning Manager
    conditioning_manager = ConditioningManager(cfg.conditioning)

    # Noise Scheduler
    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)

    # Main Model (UNet)
    latent_channels = tokenizer_config["latent_channels"]
    model = get_model(cfg, latent_channels, conditioning_manager, cfg.image_size // cfg.image_tokenizer.spatial_compression)

    # --- Load Trained Model Checkpoint ---
    # Determine the checkpoint path: Use eval.checkpoint_dir if provided, otherwise default to train.resume_model
    checkpoint_folder = cfg.train.resume_model
    model_weights_path = os.path.join(checkpoint_folder , "model.safetensors")
    logger.info(f"Loading model weights from: {model_weights_path}")
    state_dict = load_file(model_weights_path, device="cpu") # Load to CPU first

    # Load the state dictionary into the model
    # Check for potential prefixes if the model was saved differently (e.g., compiled)
    # Example: Adjust keys if they start with '_orig_mod.'
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     name = k.replace("_orig_mod.", "") if "_orig_mod." in k else k
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict) 
    
    # If accelerate saved the unwrapped model directly, prefix adjustment might not be needed:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error(f"Error loading state dict, possibly due to missing/unexpected keys or prefixes: {e}")
        logger.info("Attempting to load with strict=False (may ignore some keys)")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    logger.info("Model weights loaded successfully.")

    # --- Prepare Components with Accelerator ---
    # We still prepare for device placement, even if model is loaded manually before
    model, vae, conditioning_manager = accelerator.prepare(model, vae, conditioning_manager)
    
    # Extract the raw model if it was wrapped (e.g., by DDP after prepare)
    # Although less likely if loading state dict *before* prepare, it's safer
    model_unwrapped = extract_model_from_parallel(model)
    vae_unwrapped = extract_model_from_parallel(vae)

    # Sampler
    sampler = samplers.Sampler(
        vae_unwrapped, 
        noise_scheduler, 
        cfg.seed, 
        cfg.image_tokenizer.spatial_compression, 
        latent_channels, 
        cfg.model.noise_steps
    )

    # --- Prepare Data ---
    logger.info("Setting up data loader for FID...")
    _, val_dataloader = data.get_dataloaders(
        cfg.data.type,
        cfg,
        vae=vae_unwrapped,
        hmwm_train_dir=cfg.data.hmwm_train_dir,
        hmwm_val_dir=cfg.data.hmwm_val_dir,
        # ... include all necessary arguments for get_dataloaders ...
        coco_train_imgs=cfg.data.get('coco_train_imgs'), # Use .get for safety
        coco_val_imgs=cfg.data.get('coco_val_imgs'),
        coco_train_ann=cfg.data.get('coco_train_ann'),
        coco_val_ann=cfg.data.get('coco_val_ann'),
        image_size=cfg.image_size,
        train_batch_size=gen_batch_size,
        val_batch_size=gen_batch_size,
        conditioning_type=cfg.conditioning.type,
        conditioning_manager=conditioning_manager,
        num_past_frames=cfg.conditioning.get('num_past_frames'),
        num_future_frames=cfg.conditioning.get('num_future_frames'),
        val_stride=1
        # IMPORTANT: Do not shuffle for FID
    )
    
    val_dataloader = accelerator.prepare(val_dataloader)

    # --- Create Temporary Directories for Real and Generated Frames ---
    real_dir = tempfile.mkdtemp()
    fake_dir = tempfile.mkdtemp()
    logger.info(f"Temporary directory for real frames: {real_dir}")
    logger.info(f"Temporary directory for fake frames: {fake_dir}")

    # --- Process Data and Generate Videos ---
    saved_real_count = 0
    saved_fake_count = 0

    num_cond_frames = cfg.conditioning.get('num_past_frames', 1)
    num_gen_frames = cfg.conditioning.get('num_future_frames', 8)
    logger.info(f"Expecting {num_cond_frames} conditioning frames and {num_gen_frames} generated/ground truth frames.")

    num_fid_samples = len(val_dataloader) # Standard for FID50k
    logger.info(f"Starting generation and saving real/fake frames for {num_fid_samples} samples...")
    pbar = tqdm.tqdm(total=num_fid_samples, disable=not accelerator.is_main_process)
    
    try:
        val_iter = iter(val_dataloader) 
        while saved_fake_count < num_fid_samples:
            try:
                batch = next(val_iter)
            except StopIteration:
                logger.warning("Validation dataloader exhausted before reaching num_fid_samples. FID will be calculated on fewer samples.")
                num_fid_samples = min(saved_real_count, saved_fake_count) # Adjust target
                break

            # current_batch_size = batch['pixel_values'].shape[0] 
            # if saved_fake_count + current_batch_size > num_fid_samples:
            #      keep_count = num_fid_samples - saved_fake_count
            #      for key in batch:
            #          if isinstance(batch[key], torch.Tensor) and batch[key].shape[0] == current_batch_size:
            #              batch[key] = batch[key][:keep_count]
            #          elif isinstance(batch[key], list) and len(batch[key]) == current_batch_size: # Handle lists if present in batch
            #              batch[key] = batch[key][:keep_count]
            #      current_batch_size = keep_count
            
            # if current_batch_size == 0: continue # Skip if batch becomes empty

            # --- Save Real Frames (Ground Truth Continuation) ---
            # Adapt this based on your dataloader's output structure!
            real_videos_batch = batch['future_frames'] # B, T, C, H, W
            batch_size = real_videos_batch.shape[0]
            gt_frames_batch = real_videos_batch[:,:, 0] # B, num_gen, C, H, W
            
            gt_frames_batch_pil = (gt_frames_batch * 0.5 + 0.5).clamp(0, 1)
            gt_frames_batch_pil = (gt_frames_batch_pil * 255).byte().cpu().numpy()

            for i in range(batch_size):
                video_idx = saved_real_count + 1
                frame_img = Image.fromarray(gt_frames_batch_pil[i].transpose(1, 2, 0))
                frame_img.save(os.path.join(real_dir, f"vid{video_idx:05d}_frame{0:03d}.png"))
                saved_real_count += 1
            
            # --- Generate Fake Videos ---
            # **Modify this call according to your sampler's API**
            with torch.no_grad():
                 # Ensure batch contains the right conditioning data
                 # dtype should match the loaded model's parameters
                 model_dtype = next(model_unwrapped.parameters()).dtype 
                 sample_videos_pil, _ = sampler.sample_video( 
                         cfg=cfg, 
                         batch=batch,
                         vae=vae_unwrapped,
                         accelerator=accelerator, 
                         model=model_unwrapped,
                         guidance_scale=cfg.model.cfg_scale,
                         dtype=model_dtype, 
                         n_samples=batch_size,
                         device=device,
                         use_progress_bar=False
                     )
                 # sample_videos_pil should be list[list[PIL.Image]] [batch_size, num_gen_frames]

            # --- Save Fake Frames ---
            for i in range(batch_size):
                video_idx = saved_fake_count + 1
                frame_img = sample_videos_pil[i][0]
                frame_img.save(os.path.join(fake_dir, f"vid{video_idx:05d}_frame{0:03d}.png"))
                saved_fake_count += 1

            if accelerator.is_main_process:
                pbar.update(1)
                
    except Exception as e:
        logger.error(f"Error during generation/saving: {e}")
        # Optionally re-raise e
    finally:
        if accelerator.is_main_process:
            pbar.close()

    # Ensure all processes are finished writing before calculating FID
    accelerator.wait_for_everyone()

    # --- Calculate FID ---
    if accelerator.is_main_process:
        logger.info("Calculating FID score...")
        # Recalculate final_num_samples based on actual files saved if counts mismatched
        real_files = [f for f in os.listdir(real_dir) if f.endswith('.png')]
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.png')]
        num_real_videos_saved = len(set(f.split('_')[0] for f in real_files))
        num_fake_videos_saved = len(set(f.split('_')[0] for f in fake_files))
        
        if num_real_videos_saved != num_fake_videos_saved:
             logger.warning(f"Number of actual saved real ({num_real_videos_saved}) and fake ({num_fake_videos_saved}) videos differ. FID might be inaccurate.")
             final_num_samples = min(num_real_videos_saved, num_fake_videos_saved)
             logger.warning(f"Calculating FID based on {final_num_samples} pairs.")
        elif num_real_videos_saved < num_fid_samples:
             logger.warning(f"Fewer samples saved ({num_real_videos_saved}) than targeted ({num_fid_samples}).")
             final_num_samples = num_real_videos_saved
        else:
            final_num_samples = num_fid_samples # Use target if counts match and >= target

        if final_num_samples == 0:
            logger.error("No samples were saved correctly. Cannot calculate FID.")
            fid_score = float('nan')
        else:
            try:
                fid_score = fid.compute_fid(
                    real_dir,
                    fake_dir,
                    mode="clean", 
                    num_workers=4, 
                    batch_size=fid_batch_size,
                    device=device,
                    num_gen=final_num_samples,
                    verbose=True
                )
                logger.info(f"FID score ({final_num_samples} samples): {fid_score}")
            except Exception as e:
                logger.error(f"FID calculation failed: {e}")
                fid_score = float('nan')
            finally:
                 # --- Clean Up Temporary Directories ---
                 logger.info("Cleaning up temporary directories...")
                 try:
                     shutil.rmtree(real_dir)
                     shutil.rmtree(fake_dir)
                     logger.info("Cleanup complete.")
                 except Exception as e:
                     logger.error(f"Error during cleanup: {e}")

        # Log or save the result
        output_dir = Path(cfg.val.get("output_dir", Path(checkpoint_folder) / "fid_results")) # Default output dir if not set
        output_dir.mkdir(parents=True, exist_ok=True)
        results_log_path = output_dir / f"fid_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(results_log_path, "w") as f:
             f.write(f"FID Score ({final_num_samples} samples): {fid_score}\n")
             f.write(f"Checkpoint Folder: {checkpoint_folder}\n")
             f.write(f"Model Weights: {model_weights_path}\n")
             # f.write(f"Config used: {OmegaConf.to_yaml(cfg)}\n") # Keep this for full reproducibility
        logger.info(f"FID result saved to: {results_log_path}")

    else:
        pass 

    logger.info("FID calculation script finished.")
    logger.info(f"FID Score: {fid_score}")

if __name__ == "__main__":
    main()