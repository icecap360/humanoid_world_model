import torch
import torch.nn as nn
import einops as eo
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from flash_attn import flash_attn_func
import xformers.ops as xops
import torch.functional as F
from models.common_blocks import SinusoidalPosEmb, Attention, get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model import ImageModel

class ContextDrop(nn.Module):
    def __init__(self, cfg_prob=0.0, context_dim=512):
        super().__init__()
        self.context_dim = context_dim
        self.cfg_prob = cfg_prob
        use_cfg_embedding = cfg_prob > 0
        if use_cfg_embedding:
            self.empty_context = nn.Parameter(torch.zeros(context_dim))

    def context_drop(self, context, batch_size, tokenizer, use_cfg, force_drop_ids=False):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids == False and use_cfg:
            context =  tokenizer.get_embeddding(context)
            context = context.unsqueeze(1)
            drop_ids = torch.rand(batch_size, device=context.device) < self.cfg_prob
            context[drop_ids, :] = self.empty_context.to(context.device)
        elif force_drop_ids == False and use_cfg == False:
            context =  tokenizer.get_embeddding(context)
            context = context.unsqueeze(1)
        elif force_drop_ids == True:
            # drop_ids = torch.ones(batch_size,) # force_drop_ids == 1
            context = self.empty_context.repeat(batch_size,1)
        return context

    def forward(self, context, train, batch_size, tokenizer, use_cfg=False):
        force_drop_ids = context is None
        context = self.context_drop(context, batch_size, tokenizer, use_cfg, force_drop_ids)
        return context

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    '''
    adaLN block from Facebook's DiT
    https://github.com/facebookresearch/DiT/blob/main/models.py
    '''
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT. Based off of Facebook's DiT
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiTImgModel(ImageModel):
    '''
    Built based off of Facebook's DiT
    https://github.com/facebookresearch/DiT/blob/main/models.py
    '''
    def __init__(
        self,
        conditioning_manager,
        context_dim=512,
        input_size=32,
        patch_size=2,
        conditioning = 'text',
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cfg_prob=0.1,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.conditioning_manager = conditioning_manager
        self.conditioning = conditioning
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = nn.Sequential(
            SinusoidalPosEmb(hidden_size ),
            nn.Linear(hidden_size , hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.cfg_prob = cfg_prob
        self.y_embedder = nn.Sequential(
            nn.Linear(context_dim, hidden_size)
            )
        self.empty_context = nn.Parameter(torch.zeros(context_dim))
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder[1].weight, std=0.02)
        nn.init.normal_(self.t_embedder[1].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, batch, timesteps, device, use_cfg=False):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        latents, context = self.extract_fields_imgs(batch, use_cfg, device)
        B, C, H, W = latents.shape
        device = latents.device
        latents = self.x_embedder(latents) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        timesteps = self.t_embedder(timesteps)                   # (N, D)
        context = self.y_embedder(context).squeeze(1)    # (N, D)
        c = timesteps + context                                # (N, D)
        for block in self.blocks:
            latents = block(latents, c)                      # (N, T, D)
        latents = self.final_layer(latents, c)                # (N, T, patch_size ** 2 * out_channels)
        latents = self.unpatchify(latents)                   # (N, out_channels, H, W)
        return latents
    

if __name__ == '__main__':
    # Create a dummy conditioning manager.
    from conditioning import ConditioningManager
    from torch import optim
    class temp_conditioning_cfg:
        type = 'text'
        text_tokenizer = 't5'
    conditioning_manager = ConditioningManager(temp_conditioning_cfg())

    # Instantiate the model.
    # For a quick unit test we can use a shallow network (e.g., depth=2).
    model = DiTImgModel(
        conditioning_manager,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=2,         # shallow network for testing
        num_heads=16,
        mlp_ratio=4.0,
        cfg_prob=0.1,
        learn_sigma=False
    )

    # Set model to training mode.
    model.train()

    # Define a simple optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Random batch of 4 images of shape (channels, height, width) = (4, 32, 32)
    x = torch.randn(4, 4, 32, 32)
    # Random timesteps as integers, one per image.
    t = torch.randint(0, 1000, (4,1))
    # Random class labels (assuming 10 classes for our dummy tokenizer).
    y = conditioning_manager.get_module()['text'].tokenize(['hello', 'its me', 'what a time', 'to be alive']) # torch.randint(0, 10, (4,))
    # Dummy target: same shape as output from DiT (should be (4, 4, 32, 32) since learn_sigma=False)
    target = torch.randn(4, 4, 32, 32)

    # Run a few training steps.
    for step in range(150):
        optimizer.zero_grad()

        # Forward pass.
        out = model(x, t, y)
        loss = loss_fn(out, target)
        
        # Backward and update.
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}, Loss: {loss.item()}")