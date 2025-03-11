import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from .unet_attention import SpatialTransformer

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, output_channels, time_embed_dim, act, add_downsample=True, groups=32, eps=1e-5):
        super().__init__()
        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(in_channels, output_channels, time_embed_dim, groups=groups, act=act, eps=eps),
            ResnetBlock(output_channels, output_channels, time_embed_dim, groups=groups, act=act, eps=eps)])

        self.add_downsample = add_downsample
        if self.add_downsample:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, input_tensor, time_embed, context=None):
        x = input_tensor
        for i,block in enumerate(self.resnet_blocks):
            x = block(x, time_embed)
        if self.add_downsample:
            x = self.downsample(x)
        return x

class UNetDownBlockCrossAttn(nn.Module):
    def __init__(self, in_channels, output_channels, time_embed_dim, act, add_downsample=True, groups=32, eps=1e-5, context_dim=None, num_heads=8):
        super().__init__()
        self.resnet_block1 = ResnetBlock(in_channels, output_channels, time_embed_dim, groups=groups, act=act, eps=eps)
        self.resnet_block2 = ResnetBlock(output_channels, output_channels, time_embed_dim, groups=groups, act=act, eps=eps)
        self.attn_block = SpatialTransformer(
                            output_channels, num_heads, output_channels // num_heads, depth=1, context_dim=context_dim
                        )

        self.add_downsample = add_downsample
        if self.add_downsample:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, input_tensor, time_embed, context=None):
        x = input_tensor
        x = self.resnet_block1(x, time_embed)
        x = self.attn_block(x, context)
        x = self.resnet_block2(x, time_embed)
        if self.add_downsample:
            x = self.downsample(x)
        return x
    
class UNetMidBlock(nn.Module):
    def __init__(self, in_channels, output_channels, time_embed_dim, act,groups=32, eps=1e-5, context_dim=None, num_heads=8):
        super().__init__()
        self.resnet_block1 = ResnetBlock(in_channels, output_channels, time_embed_dim, act=act, groups=groups, eps=eps)
        self.attn_block = SpatialTransformer(
                    output_channels, num_heads, output_channels // num_heads, depth=1, context_dim=context_dim
                )
        self.resnet_block2 = ResnetBlock(output_channels, output_channels, time_embed_dim, act=act, groups=groups, eps=eps)
        
    def forward(self, input_tensor, time_embed, context=None):
        x = input_tensor
        x = self.resnet_block1(x, time_embed)
        x = self.attn_block(x, context)
        x = self.resnet_block2(x, time_embed)
        return x
    
class UNetUpBlock(nn.Module):
    def __init__(self, skip_channels, down_channels, output_channels, time_embed_dim, act, add_upsample=True, groups=32, eps=1e-5):
        super().__init__()
        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(skip_channels + down_channels, output_channels, time_embed_dim,act=act,  groups=groups, eps=eps),
            ResnetBlock(output_channels, output_channels, time_embed_dim, groups=groups, act=act, eps=eps)])

        self.add_upsample = add_upsample
        if self.add_upsample:
            self.upsample_conv = nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1)

    def forward(self, from_down, from_up, time_embed, context=None):
        if self.add_upsample:
            dtype = from_down.dtype
            from_down = F.interpolate(from_down, scale_factor=2.0, mode="bilinear")
            if dtype == torch.bfloat16:
                from_down = from_down.to(dtype)
            from_down = self.upsample_conv(from_down)
        x = torch.cat([from_down, from_up], dim=1)
        for block in self.resnet_blocks:
            x = block(x, time_embed)
        return x

class UNetUpBlockCrossAttn(nn.Module):
    def __init__(self, skip_channels, down_channels, output_channels, time_embed_dim, act, add_upsample=True, groups=32, eps=1e-5, context_dim=None, num_heads=8):
        super().__init__()
        self.resnet_block1 = ResnetBlock(skip_channels + down_channels, output_channels, time_embed_dim,act=act,  groups=groups, eps=eps)
        self.resnet_block2 = ResnetBlock(output_channels, output_channels, time_embed_dim, groups=groups, act=act, eps=eps)
        self.attn_block = SpatialTransformer(
                    output_channels, num_heads, output_channels // num_heads, depth=1, context_dim=context_dim
                )
        self.add_upsample = add_upsample
        if self.add_upsample:
            self.upsample_conv = nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1)

    def forward(self, from_down, from_up, time_embed, context=None):
        if self.add_upsample:
            dtype = from_down.dtype
            from_down = F.interpolate(from_down, scale_factor=2.0, mode="bilinear")
            if dtype == torch.bfloat16:
                from_down = from_down.to(dtype)
            from_down = self.upsample_conv(from_down)
        x = torch.cat([from_down, from_up], dim=1)
        x = self.resnet_block1(x, time_embed)
        x = self.attn_block(x, context)
        x = self.resnet_block2(x, time_embed)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, act, groups=32, eps=1e-5, p_drop=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups=groups
        self.eps = eps

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.act = act
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 =  nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=p_drop)
        # notice how rechannel is just a conv, no activation or norm
        if self.in_channels != self.out_channels:
            self.rechannel = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )

        self.time_embed_dim = time_embed_dim
        self.time_scale_shift = ScaleShift(time_embed_dim, out_channels)

    def forward(self, input_tensor, time_embed):
        x = input_tensor
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        # notice how we embed timestep after the first convolution
        # notice how norm2 does not undo the effect of the time_scale_shift, as it preserves relative distances but not absolute differences
        # notice how time_scale_shift is applied after norm2, this was discovered in "Diffusion beats GAN" paper
        x = self.norm2(x)
        x = self.time_scale_shift(x, time_embed)
        x = self.dropout(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            input_tensor = self.rechannel(input_tensor)
        return x + input_tensor

class ScaleShift(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # notice how GELU worked better then SILU here
        self.act = nn.GELU()
        self.scale_proj = nn.Linear(self.in_dim, self.out_dim)
        self.shift_proj = nn.Linear(self.in_dim, self.out_dim)
    def forward(self, x, x_condition):
        # notice how the activation is applied first
        x_condition = self.act(x_condition)
        # notice how there is only 1 linear layer that applies the projection
        scale = self.scale_proj(x_condition)
        shift = self.shift_proj(x_condition)
        # notice how you rescale x by scale+1, because NN typically output close to 0 
        return x*(scale[:,:, None, None]+1) + shift[:, :, None, None]
    
        
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t, dtype=torch.float32):
        # from lucidrains
        # device = t.device
        # half_dim = self.dim // 2
        # emb = math.log(self.theta) / (half_dim - 1)
        # emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # emb = t[:, None] * emb[None, :]
        # emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # notice how we interleave the sin and cos, some repos simply concatenate them
        time_indices = torch.arange(0, self.dim, 2)
        denominator = torch.exp(math.log(self.theta) * 2* time_indices / self.dim)
        denominator = denominator.to(t.device)
        emb = torch.zeros((t.shape[0], self.dim), device=t.device)
        emb[:, 0::2] = torch.sin(t / denominator)
        emb[:, 1::2] = torch.cos(t / denominator)
        return emb
    
class AttnBlock(nn.Module):
    '''Taken from Stable Diffusion'''
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

# class AttentionBlock(nn.Module):
#     """
#     An attention block that allows spatial positions to attend to each other.
#     Originally ported from here, but adapted to the N-d case.
#     https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
#     """

#     def __init__(
#         self,
#         channels,
#         num_heads=1,
#         num_head_channels=-1,
#         use_checkpoint=False,
#         use_new_attention_order=False,
#     ):
#         super().__init__()
#         self.channels = channels
#         if num_head_channels == -1:
#             self.num_heads = num_heads
#         else:
#             assert (
#                 channels % num_head_channels == 0
#             ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
#             self.num_heads = channels // num_head_channels
#         self.use_checkpoint = use_checkpoint
#         self.norm = nn.GroupNorm(32, channels)
#         self.qkv = nn.Conv1d(1, channels, channels * 3, 1)
#         self.attention = QKVAttention(self.num_heads)

#         self.proj_out = nn.Conv1d(1, channels, channels, 1)
#         for p in self.proj_out.parameters():
#             p.detach().zero_()

#     def forward(self, x):
#         return self._forward(x) 

#     def _forward(self, x):
#         b, c, *spatial = x.shape
#         x = x.reshape(b, c, -1)
#         qkv = self.qkv(self.norm(x))
#         h = self.attention(qkv)
#         h = self.proj_out(h)
#         return (x + h).reshape(b, c, *spatial)

# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention and splits in a different order.
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv):
#         """
#         Apply QKV attention.
#         :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.chunk(3, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = th.einsum(
#             "bct,bcs->bts",
#             (q * scale).view(bs * self.n_heads, ch, length),
#             (k * scale).view(bs * self.n_heads, ch, length),
#         )  # More stable with f16 than dividing afterwards
#         weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
#         return a.reshape(bs, -1, length)

#     @staticmethod
#     def count_flops(model, _x, y):
#         return count_flops_attn(model, _x, y)
