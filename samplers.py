import torch
from einops import rearrange
from configs import _UINT8_MAX_F
import numpy as np
import tqdm 
import PIL 

class Sampler:
    def __init__(self, vae, scheduler, seed, spatial_compression, latent_channels, num_inference_steps):
        self.vae = vae
        self.seed = seed
        self.scheduler = scheduler
        self.spatial_compression = spatial_compression
        self.latent_channels = latent_channels
        self.num_inference_steps = num_inference_steps
        
    def sample(self, unet, img_size, in_channels, dtype=torch.float32, device='cuda'):
        batch_size = 16
        if isinstance(img_size, int):
            sample_height = img_size
            sample_width = img_size
            image_shape = (
                batch_size,
                in_channels,
                img_size,
                img_size,
            )
        else:
            sample_height = img_size[0]
            sample_width = img_size[1]
            image_shape = (batch_size, in_channels, *img_size)
        
        generator=torch.Generator(device=device).manual_seed(self.seed)
        latents = torch.randn((batch_size, self.latent_channels, sample_height//self.spatial_compression, sample_width//self.spatial_compression), dtype=dtype, device=device, generator=generator)
        
        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)
        for t in self.progress_bar(timesteps):
            # 1. predict noise model_output
            # model_output = unet(latents, t).sample
            batch = {
                'noisy_latents': latents,
            }
            model_output = unet(batch, t)

            # 2. compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(model_output, t, latents, generator=generator).prev_sample
        
        vae = self.vae.module if hasattr(self.vae, "module") else self.vae
        pred_img = vae.decode(latents.to(torch.bfloat16))
        pred_img = decode_img(pred_img)
        pred_img = [PIL.Image.fromarray(s) for s in pred_img]
        return pred_img
    def sample_text(self, model, img_size, in_channels, text_tokenizer, prompt_file, guidance_scale, dtype=torch.float32, device='cuda'):
        batch_size = 16
        if isinstance(img_size, int):
            sample_height = img_size
            sample_width = img_size
            image_shape = (
                batch_size,
                in_channels,
                img_size,
                img_size,
            )
        else:
            sample_height = img_size[0]
            sample_width = img_size[1]
            image_shape = (batch_size, in_channels, *img_size)
        
        with open(prompt_file) as f:
            user_prompts = f.readlines()
            prompts = []
            # extend the prompt cyclically if batch_size > len(user_prompts)
            for i in range(batch_size):
                prompts.append(user_prompts[i % len(user_prompts)])
            prompts = text_tokenizer.tokenize(prompts)        
        
        generator=torch.Generator(device=device).manual_seed(self.seed)
        latents = torch.randn((batch_size, self.latent_channels, sample_height//self.spatial_compression, sample_width//self.spatial_compression), dtype=dtype, device=device, generator=generator)
        
        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)
        for t in self.progress_bar(timesteps):
            # 1. predict noise model_output
            # model_output = unet(latents, t).sample
            t = t.reshape(1,1,1,1)
            batch = {
                'noisy_latents': latents,
                'captions': prompts
            }
            pred_cond = model(batch, t, device, use_cfg=False)
            batch.pop('captions')
            pred_uncond = model(batch, t, device, use_cfg=False)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # 2. compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(pred, t, latents, generator=generator).prev_sample
        
        vae = self.vae.module if hasattr(self.vae, "module") else self.vae
        pred_img = vae.decode(latents.to(torch.bfloat16))
        pred_img = decode_img(pred_img)
        pred_img = [PIL.Image.fromarray(s) for s in pred_img]
        return pred_img
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm.tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm.tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

def decode_img(preds):
    output_imgs = (preds.float() + 1.0) / 2.0
    output_imgs = rearrange(output_imgs, 'b c h w -> b h w c')
    output_imgs = output_imgs.clamp(0, 1).cpu().numpy()
    output_imgs = output_imgs * _UINT8_MAX_F + 0.5
    output_imgs = output_imgs.astype(np.uint8)
    return output_imgs

def sample_hf_pipeline(pipeline, seed):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=16,
        generator=torch.Generator(device='cpu').manual_seed(seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    return images