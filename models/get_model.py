from .unet import UNet
from .dit_imgs import DiTImgModel
from diffusers import UNet2DModel
from .dit_video import VideoDiTModel, VideoUViTModel

def get_model(cfg, latent_channels, input_size, conditioning_manager=None):
    if 'unet' in cfg.model.type:
        model = UNet(latent_channels, 
                latent_channels, 
                cfg.model.unet_blocks, 
                256, 
                conditioning_manager=conditioning_manager,
                cfg_prob=cfg.model.cfg_prob,
                unet_attention_resolutions=cfg.model.unet_attention_resolutions)
        return model
    elif cfg.model.type=='hf':
        model = UNet2DModel(
            sample_size=cfg.model.image_size,  # the target image resolution
            in_channels=latent_channels,  # the number of input channels, 3 for RGB images
            out_channels=latent_channels,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels= (224, 224*2, 224*3, 224*4), # (64, 64, 128, 128, 256, 256), # (128, 128, 256, 256, 512, 512), # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                # "DownBlock2D",
                # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                # "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                # "UpBlock2D",
                # "UpBlock2D",
            ),
        )
        return model
    elif cfg.model.type.lower() == 'dit_xl_2':
        return DiTImgModel(depth=28, hidden_size=1152, patch_size=2, num_heads=16, 
                   in_channels=latent_channels,
                   input_size=input_size,
                   cfg_prob=cfg.model.cfg_prob, 
                   conditioning_manager=conditioning_manager)
    elif cfg.model.type.lower() == 'dit_s_2':
        return DiTImgModel(depth=12, hidden_size=384, patch_size=2, num_heads=6, in_channels=latent_channels,
        input_size=input_size,
        cfg_prob=cfg.model.cfg_prob, 
        conditioning_manager=conditioning_manager)
    elif cfg.model.type.lower() == 'video_dit':
        dim_spatial=cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoDiTModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time
        )
    elif cfg.model.type.lower() == 'video_uvit':
        dim_spatial=cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoUViTModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time
        )
    else:
        raise Exception('Unknown model type')
