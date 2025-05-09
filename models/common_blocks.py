import math
import time
from typing import Dict, List, Optional, Tuple, Union

import einops as eo
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
from timm.models.vision_transformer import Mlp, PatchEmbed
from torch.nn.functional import scaled_dot_product_attention


class QKV(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Normalization operates over the last dimension (head_dim)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        qkv = self.qkv(x)
        qkv = rearrange(
            qkv,
            "... (qkv h d) -> ... qkv h d",
            qkv=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q, k = self.q_norm(q), self.k_norm(k)

        # Reshape to [B, num_heads, N, head_dim]
        q = rearrange(q, "... n h d-> ... h n d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "... n h d-> ... h n d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "... n h d-> ... h n d", h=self.num_heads, d=self.head_dim)

        return q, k, v


class KV(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # Normalization operates over the last dimension (head_dim)
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        qkv = self.kv(x)
        qkv = rearrange(
            qkv,
            "... (qkv h d) -> ... qkv h d",
            qkv=2,
            h=self.num_heads,
            d=self.head_dim,
        )

        q, k = qkv[..., 0, :, :], qkv[..., 1, :, :]
        k = self.k_norm(k)

        # Reshape to [B, num_heads, N, head_dim]
        q = rearrange(q, "... n h d-> ... h n d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "... n h d-> ... h n d", h=self.num_heads, d=self.head_dim)
        return q, k


class Q(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # Normalization operates over the last dimension (head_dim)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        qkv = self.q(x)
        qkv = rearrange(
            qkv,
            "... (qkv h d) -> ... qkv h d",
            qkv=1,
            h=self.num_heads,
            d=self.head_dim,
        )

        q = qkv[..., 0, :, :]
        q = self.q_norm(q)

        # Reshape to [B, num_heads, N, head_dim]
        q = rearrange(q, "... n h d-> ... h n d", h=self.num_heads, d=self.head_dim)

        return q


class JointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_causal=False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.p_attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

    def forward(self, all_qkv):
        qs, ks, vs = [], [], []
        for qkv in all_qkv:
            qs.append(qkv[0])
            ks.append(qkv[1])
            vs.append(qkv[2])

        qs, pack_info = pack(qs, "b h * d")
        ks, _ = pack(ks, "b h * d")
        vs, _ = pack(vs, "b h * d")

        out = scaled_dot_product_attention(
            qs,
            ks,
            vs,
            dropout_p=self.p_attn_drop if self.training else 0.0,
            is_causal=self.is_causal,
        )

        out = rearrange(
            out, "... h n d-> ... n (h d)", h=self.num_heads, d=self.head_dim
        )
        out = self.proj_drop(self.proj(out))

        out = unpack(out, pack_info, "b * d")
        return out


class JointFactorizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_causal=False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.p_attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

    def forward(self, all_qkv):
        qs, ks, vs = [], [], []
        for qkv in all_qkv:
            qs.append(qkv[0])
            ks.append(qkv[1])
            vs.append(qkv[2])

        qs, pack_info = pack(qs, "b x h * d")
        ks, pack_info = pack(ks, "b x h * d")
        vs, pack_info = pack(vs, "b x h * d")

        out = scaled_dot_product_attention(
            qs,
            ks,
            vs,
            dropout_p=self.p_attn_drop if self.training else 0.0,
            is_causal=self.is_causal,
        )

        out = rearrange(
            out, "... h n d-> ... n (h d)", h=self.num_heads, d=self.head_dim
        )
        out = self.proj_drop(self.proj(out))

        out = unpack(out, pack_info, "b x * d")
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_factor=4, p_dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * hidden_factor)
        self.act = act
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear2 = nn.Linear(dim * hidden_factor, dim)

    def forward(self, x):
        x = self.dropout(self.act(self.linear1(x)))
        x = self.linear2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_type="torch",
        norm_layer: nn.Module = nn.LayerNorm,
        is_causal=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_type = attn_type
        assert attn_type in ["torch", "flash", "xformers", "naive"]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Normalization operates over the last dimension (head_dim)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute qkv and reshape directly to minimize operations
        if self.attn_type == "flash":
            # For flash attention, reshape directly to [B, N, num_heads, 3, head_dim]
            # This avoids the intermediate reshape and permute operations
            qkv = self.qkv(x).reshape(B, N, self.num_heads, 3, self.head_dim)
            q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
            # Apply normalization
            q, k = self.q_norm(q), self.k_norm(k)
            # Flash attention expects [B, N, num_heads, head_dim]
            x_attn = flash_attn_func(q, k, v)

        elif self.attn_type == "xformers":
            import xformers.ops as xops

            # Reshape to [B, N, 3, num_heads, head_dim] for xformers
            # notice the order of the reshape. It matters and is correct because torch uses row-major C-style order, so the last dimension (head_dim) changes the fastest, while the qkv dimension (which is 3) changes the slowest
            # e.g. qkv.reshape(B, N, 3, self.num_heads, self.head_dim) != qkv.reshape(B, N, self.num_heads, self.head_dim, 3).moveaxis(-1, 2)
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = self.q_norm(q), self.k_norm(k)

            # Reshape directly to [B*num_heads, N, head_dim]
            q_xf = q.reshape(B * self.num_heads, N, self.head_dim)
            k_xf = k.reshape(B * self.num_heads, N, self.head_dim)
            v_xf = v.reshape(B * self.num_heads, N, self.head_dim)

            out = xops.memory_efficient_attention(q_xf, k_xf, v_xf)
            # Reshape back to [B, N, num_heads, head_dim]
            x_attn = out.reshape(B, N, self.num_heads, self.head_dim)

        elif self.attn_type == "torch":
            # Reshape for torch's scaled_dot_product_attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = self.q_norm(q), self.k_norm(k)

            # Reshape to [B, num_heads, N, head_dim]
            q_t = q.permute(0, 2, 1, 3)
            k_t = k.permute(0, 2, 1, 3)
            v_t = v.permute(0, 2, 1, 3)

            out = torch.nn.functional.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=self.is_causal,
            )
            # Reshape to [B, N, num_heads, head_dim]
            x_attn = out.permute(0, 2, 1, 3)

        elif self.attn_type == "naive":
            # Naive implementation
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = self.q_norm(q), self.k_norm(k)

            # Reshape to [B, num_heads, N, head_dim]
            q_t = q.permute(0, 2, 1, 3)
            k_t = k.permute(0, 2, 1, 3)
            v_t = v.permute(0, 2, 1, 3)

            q_t = q_t * self.scale
            attn = torch.matmul(q_t, k_t.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v_t)

            # Reshape back to [B, N, num_heads, head_dim]
            x_attn = out.permute(0, 2, 1, 3)

        else:
            raise ValueError(f"Unknown attn_type {self.attn_type}")

        # Merge the heads and apply the final projection
        x_out = x_attn.reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


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
        return x * (scale[:, :, None, None] + 1) + shift[:, :, None, None]


class SinusoidalPosEmb(nn.Module):
    # Notice how theta=300 here, even though in LLMs they typically use theta=1000
    # notice how the flow model is trained and samp led with timesteps in tange [0,1], but the sinusoidal embedding required time indices in range [0,1000]
    # the sinusoidal position embedding requires time indices be in range [0,1000], but the flow model can be sampled at any point continuous range of t=[0,1]
    def __init__(self, dim, theta=300):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t, mul_factor=1000, dtype=torch.float32):
        # from lucidrains
        # device = t.device
        # half_dim = self.dim // 2
        # emb = math.log(self.theta) / (half_dim - 1)
        # emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # emb = t[:, None] * emb[None, :]
        # emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        # notice how we interleave the sin and cos, some repos simply concatenate them
        t = t * mul_factor
        half = self.dim // 2
        inds = torch.arange(half, device=t.device, dtype=t.dtype)
        freqs = (-math.log(self.theta) * inds / half).exp()
        embs = t[:, None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim=-1)
        embs = embs.squeeze(1)
        return embs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ActionPositionEmb(nn.Module):
    def __init__(self, len_t, hidden_dim, theta=10000):
        super().__init__()
        self.theta = float(theta)
        self.register_buffer(
            "position_encoding",
            self.rotary_position_embedding(len_t, hidden_dim, theta=self.theta),
        )

    def forward(self, x):
        return self.apply_rope_embeddings(x, self.position_encoding)

    def rotary_position_embedding(self, max_seq_len, dim, theta):
        # Calculate the angle rates based on dimension indices.
        angle_rates = 1 / torch.pow(theta, torch.arange(0, dim, 2).float() / dim)
        # Calculate the angles for each position for half of the dimensions (sine and cosine)
        angles = torch.arange(max_seq_len).unsqueeze(1) * angle_rates.unsqueeze(0)
        # Cosines and sines of the angles to get the RoPE for each position
        position_encodings = torch.stack((angles.cos(), angles.sin()), dim=2).flatten(1)
        return position_encodings

    def apply_rope_embeddings(self, embeddings, position_encodings):
        # Split the position encodings into cosines and sines
        cos_enc, sin_enc = position_encodings[..., 0::2], position_encodings[..., 1::2]
        # Apply the rotations
        embeddings[..., 0::2] = (
            embeddings[..., 0::2] * cos_enc - embeddings[..., 1::2] * sin_enc
        )
        embeddings[..., 1::2] = (
            embeddings[..., 1::2] * cos_enc + embeddings[..., 0::2] * sin_enc
        )
        return embeddings


class ActionLearnablePositionEmb(nn.Module):
    """
    Learnable positional embeddings for action tokens in **bhsd** format.

    Args
    ----
    len_t : int
        Maximum sequence length.
    hidden_dim : int
        Embedding dimension (must match the model’s hidden size).
    theta : float, optional
        Kept for API compatibility; unused in the learnable version.
    """

    def __init__(self, len_t: int, hidden_dim: int, theta: float = 10000):
        super().__init__()
        self.theta = float(theta)  # kept to avoid breaking old configs

        # Trainable table of shape [seq_len, hidden_dim]
        self.pos_embed = nn.Parameter(torch.zeros(len_t, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # ViT-style init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape **[B, H, S, D]**.

        Returns
        -------
        torch.Tensor
            Same shape as `x`, with positional offsets added.
        """
        b, h, s, d = x.shape
        if s > self.pos_embed.size(0):
            raise ValueError(
                f"Sequence length {s} exceeds maximum {self.pos_embed.size(0)}"
            )

        pos = self.pos_embed[:s].to(dtype=x.dtype)  # [S, D]
        pos = pos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
        return x + pos  # broadcast over


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 30,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        theta=10000.0,
        device="cuda",
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer(
            "seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float)
        )
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert (
            dim == dim_h + dim_w + dim_t
        ), f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.theta = float(theta)

    def generate_embeddings(
        self,
        T_H_W: torch.Size,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            T_H_W (torch.Size): Input tensor size (Time, Height, Width).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = self.theta * h_ntk_factor
        w_theta = self.theta * w_ntk_factor
        t_theta = self.theta * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        T, H, W = T_H_W
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        # apply sequence scaling in temporal dimension
        # if fps is None:  # image case
        #     assert T == 1, "T should be 1 for image batch."
        #     half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
        # else:
        half_emb_t = torch.outer(
            self.seq[:T] / self.base_fps * self.base_fps, temporal_freqs
        )

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    def forward(self, x):
        return x


class VideoPositionEmb(VideoRopePosition3DEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 30,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        theta=10000.0,
        device="cuda",
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        super().__init__(
            head_dim=head_dim,
            len_h=len_h,
            len_w=len_w,
            len_t=len_t,
            base_fps=base_fps,
            h_extrapolation_ratio=h_extrapolation_ratio,
            w_extrapolation_ratio=w_extrapolation_ratio,
            t_extrapolation_ratio=t_extrapolation_ratio,
            theta=theta,
            device=device,
        )
        self.register_buffer(
            "position_encoding",
            self.generate_embeddings(
                (len_t, len_h, len_w),
                h_ntk_factor=h_extrapolation_ratio,
                w_ntk_factor=w_extrapolation_ratio,
                t_ntk_factor=t_extrapolation_ratio,
            ),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.apply_rotary_pos_emb(x, self.position_encoding)
        return x

    def apply_rotary_pos_emb(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "bhsd",
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to `t` in one of three layouts:
          - 'sbhd': [seq, batch, heads, dim]
          - 'bshd': [batch, seq, heads, dim]
          - 'bhsd': [batch, heads, seq, dim]
        """
        assert tensor_format in (
            "sbhd",
            "bshd",
            "bhsd",
        ), f"Only formats `sbhd`, `bshd` or `bhsd` are supported, got {tensor_format}."

        max_seq_len = freqs.shape[0]
        # pick out current sequence length based on layout
        if tensor_format == "bshd":
            cur_seq_len = t.shape[1]
        elif tensor_format == "bhsd":
            cur_seq_len = t.shape[2]
        else:  # sbhd
            cur_seq_len = t.shape[0]

        assert cur_seq_len <= max_seq_len, (
            f"Rotary embeddings only supported up to length {max_seq_len}, "
            f"got {cur_seq_len}."
        )
        freqs = freqs[:cur_seq_len]  # [seq, 1, 1, dim]

        # reshape freqs to align with t
        if tensor_format == "bshd":
            # from [seq,1,1,dim] -> [1,seq,1,dim]
            freqs = freqs.transpose(0, 1)
        elif tensor_format == "bhsd":
            # from [seq,1,1,dim] -> [1,1,seq,dim]
            freqs = freqs.permute(1, 2, 0, 3)
        # if 'sbhd', leave as [seq,1,1,dim]

        # compute cos/sin once
        cos_ = torch.cos(freqs).to(t.dtype)
        sin_ = torch.sin(freqs).to(t.dtype)

        rot_dim = freqs.shape[-1]
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

        # apply: (x * cos) + (rotate_half(x) * sin)
        t_out = (t_rot * cos_) + (_rotate_half(t_rot) * sin_)
        return torch.cat((t_out, t_pass), dim=-1)


class VideoLearnedPositionEmb(VideoRopePosition3DEmb):
    """
    Learnable 3-D positional embeddings for video transformers.

    *Only* supports tensor layout **bhsd** → [batch, heads, seq, dim].
    """

    def __init__(
        self,
        *,  # enforce keyword args
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 30,  # kept for API compatibility
        **kwargs,  # ignore extra arguments
    ):
        super().__init__(
            head_dim=head_dim,
            len_h=len_h,
            len_w=len_w,
            len_t=len_t,
            base_fps=base_fps,
            **kwargs,
        )

        # Trainable grid [t, h, w, dim]  → flattened at runtime
        self.pos_embed = nn.Parameter(
            torch.zeros(len_t, len_h, len_w, head_dim)  # initialised to zero
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to `x`.

        Parameters
        ----------
        x : torch.Tensor
            Shape **[B, H, S, D]** where S = len_t · len_h · len_w.

        Returns
        -------
        torch.Tensor
            Same shape as `x`, with learnable positional offsets added.
        """
        b, h, s, d = x.shape
        pos = self.pos_embed.view(-1, d).to(x.dtype)  # [S, D]
        assert s == pos.shape[0], "sequence length mismatch"

        # Broadcast to [1, 1, S, D] → added over batch & heads
        return x + pos[None, None, :, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


class AttentionProfiler:
    def __init__(self, device: str = "cuda"):
        """
        Initialize the profiler with the specified device.

        Args:
            device: Device to run the profiling on ('cuda' or 'cpu')
        """
        self.device = device
        self.attention_types = ["naive", "torch", "xformers", "flash"]
        self.results = {}

    def _generate_inputs(
        self, batch_size: int, seq_len: int, dim: int, dtype=torch.float32
    ) -> torch.Tensor:
        """Generate random input tensors for profiling."""
        return torch.randn(batch_size, seq_len, dim, device=self.device, dtype=dtype)

    def _create_attention_module(
        self, attn_type: str, dim: int, num_heads: int, dtype=torch.float32
    ) -> Attention:
        """Create an attention module with the specified configuration."""
        return Attention(
            dim=dim, num_heads=num_heads, qkv_bias=True, attn_type=attn_type
        ).to(self.device)

    def profile_single_config(
        self,
        batch_size: int,
        seq_len: int,
        dim: int,
        num_heads: int,
        num_repeats: int = 100,
        warmup: int = 10,
        dtype=torch.float32,
    ) -> Dict[str, float]:
        """
        Profile all attention types for a single configuration.

        Args:
            batch_size: Batch size for input
            seq_len: Sequence length for input
            dim: Hidden dimension
            num_heads: Number of attention heads
            num_repeats: Number of times to repeat the forward pass for timing
            warmup: Number of warmup runs before timing

        Returns:
            Dictionary of average execution times for each attention type
        """
        results = {}
        inputs = self._generate_inputs(batch_size, seq_len, dim, dtype)

        for attn_type in self.attention_types:
            try:
                # Skip flash attention for CPU
                if attn_type == "flash" and self.device == "cpu":
                    print(f"Skipping flash attention on CPU as it's not supported")
                    continue

                model = self._create_attention_module(attn_type, dim, num_heads).to(
                    dtype
                )
                model.eval()  # Set to evaluation mode to disable dropout

                # Warmup
                for _ in range(warmup):
                    with torch.no_grad():
                        _ = model(inputs)

                # Perform timing
                start_time = time.time()
                for _ in range(num_repeats):
                    with torch.no_grad():
                        _ = model(inputs)

                # Synchronize if using CUDA
                if self.device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()
                avg_time = (end_time - start_time) / num_repeats * 1000  # Convert to ms
                results[attn_type] = avg_time

                print(
                    f"Config [B={batch_size}, L={seq_len}, D={dim}, H={num_heads}] "
                    f"- {attn_type}: {avg_time:.3f} ms"
                )

            except Exception as e:
                print(f"Error profiling {attn_type} attention: {e}")
                results[attn_type] = float("nan")

        return results

    def profile_varying_sequence_length(
        self,
        batch_size: int,
        seq_lengths: List[int],
        dim: int,
        num_heads: int,
        num_repeats: int = 100,
        dtype=torch.float32,
    ):
        """Profile with varying sequence lengths."""
        results = {attn_type: [] for attn_type in self.attention_types}

        for seq_len in seq_lengths:
            print(f"\nProfiling with sequence length {seq_len}")
            config_result = self.profile_single_config(
                batch_size, seq_len, dim, num_heads, num_repeats, dtype=dtype
            )

            for attn_type, time_ms in config_result.items():
                results[attn_type].append(time_ms)

        self.results["seq_length"] = {"seq_lengths": seq_lengths, "times": results}
        return results

    def profile_varying_hidden_dim(
        self,
        batch_size: int,
        seq_len: int,
        dims: List[int],
        num_heads_list: Optional[List[int]] = None,
        num_repeats: int = 100,
        dtype=torch.float32,
    ):
        """Profile with varying hidden dimensions."""
        if num_heads_list is None:
            # Default to dim / 64 heads
            num_heads_list = [dim // 64 for dim in dims]

        assert len(dims) == len(
            num_heads_list
        ), "dims and num_heads_list must have same length"

        results = {attn_type: [] for attn_type in self.attention_types}

        for dim, num_heads in zip(dims, num_heads_list):
            print(f"\nProfiling with dimension {dim} and {num_heads} heads")
            config_result = self.profile_single_config(
                batch_size, seq_len, dim, num_heads, num_repeats, dtype=dtype
            )

            for attn_type, time_ms in config_result.items():
                results[attn_type].append(time_ms)

        self.results["hidden_dim"] = {
            "dims": dims,
            "num_heads": num_heads_list,
            "times": results,
        }
        return results

    def profile_varying_batch_size(
        self,
        batch_sizes: List[int],
        seq_len: int,
        dim: int,
        num_heads: int,
        num_repeats: int = 100,
        dtype=torch.float32,
    ):
        """Profile with varying batch sizes."""
        results = {attn_type: [] for attn_type in self.attention_types}

        for batch_size in batch_sizes:
            print(f"\nProfiling with batch size {batch_size}")
            config_result = self.profile_single_config(
                batch_size, seq_len, dim, num_heads, num_repeats, dtype=dtype
            )

            for attn_type, time_ms in config_result.items():
                results[attn_type].append(time_ms)

        self.results["batch_size"] = {"batch_sizes": batch_sizes, "times": results}
        return results

    def profile_memory_usage(
        self,
        batch_size: int,
        seq_len: int,
        dim: int,
        num_heads: int,
        dtype=torch.float32,
    ):
        """Profile memory usage for each attention type."""
        if self.device != "cuda":
            print("Memory profiling is only available on CUDA devices")
            return {}

        memory_usage = {}
        inputs = self._generate_inputs(batch_size, seq_len, dim, dtype)

        for attn_type in self.attention_types:
            try:
                # Skip flash attention for CPU
                if attn_type == "flash" and self.device == "cpu":
                    continue

                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                model = self._create_attention_module(attn_type, dim, num_heads).to(
                    dtype
                )

                # Forward pass
                with torch.no_grad():
                    _ = model(inputs)

                # Get memory stats
                memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                memory_usage[attn_type] = memory_used

                print(
                    f"Config [B={batch_size}, L={seq_len}, D={dim}, H={num_heads}] "
                    f"- {attn_type}: {memory_used:.2f} MB"
                )

            except Exception as e:
                print(f"Error profiling memory for {attn_type} attention: {e}")
                memory_usage[attn_type] = float("nan")

        self.results["memory"] = memory_usage
        return memory_usage

    def plot_results(self, save_path: Optional[str] = None):
        """Plot the profiling results."""
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Plot 1: Sequence Length vs. Time
        if "seq_length" in self.results:
            data = self.results["seq_length"]
            seq_lengths = data["seq_lengths"]
            times = data["times"]

            for attn_type, time_list in times.items():
                if any(not np.isnan(t) for t in time_list):
                    axs[0].plot(seq_lengths, time_list, marker="o", label=attn_type)

            axs[0].set_xlabel("Sequence Length")
            axs[0].set_ylabel("Time (ms)")
            axs[0].set_title("Attention Performance vs. Sequence Length")
            axs[0].legend()
            axs[0].grid(True)

        # Plot 2: Hidden Dimension vs. Time
        if "hidden_dim" in self.results:
            data = self.results["hidden_dim"]
            dims = data["dims"]
            times = data["times"]

            for attn_type, time_list in times.items():
                if any(not np.isnan(t) for t in time_list):
                    axs[1].plot(dims, time_list, marker="o", label=attn_type)

            axs[1].set_xlabel("Hidden Dimension")
            axs[1].set_ylabel("Time (ms)")
            axs[1].set_title("Attention Performance vs. Hidden Dimension")
            axs[1].legend()
            axs[1].grid(True)

        # Plot 3: Batch Size vs. Time
        if "batch_size" in self.results:
            data = self.results["batch_size"]
            batch_sizes = data["batch_sizes"]
            times = data["times"]

            for attn_type, time_list in times.items():
                if any(not np.isnan(t) for t in time_list):
                    axs[2].plot(batch_sizes, time_list, marker="o", label=attn_type)

            axs[2].set_xlabel("Batch Size")
            axs[2].set_ylabel("Time (ms)")
            axs[2].set_title("Attention Performance vs. Batch Size")
            axs[2].legend()
            axs[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Results saved to {save_path}")
        else:
            plt.show()

    def summarize_results(self):
        """Print a summary of the profiling results."""
        print("\n===== ATTENTION PROFILING SUMMARY =====")

        if "memory" in self.results:
            print("\nMemory Usage (MB):")
            # memory_data = self.


if __name__ == "__main__":
    profiler = AttentionProfiler("cuda")

    print("vary seq len", "torch.float32")
    print(
        profiler.profile_varying_sequence_length(
            16, [256, 512, 768, 1024, 2048], 512, 8, dtype=torch.float32
        )
    )

    print("vary seq len", "torch.bfloat16")
    print(
        profiler.profile_varying_sequence_length(
            16, [256, 512, 768, 1024, 2048], 512, 8, dtype=torch.bfloat16
        )
    )

    print("vary hidden dim", "torch.float32")
    print(
        profiler.profile_varying_hidden_dim(
            16, 512, [256, 512, 768, 1024, 2048], [8, 8, 8, 8, 8], dtype=torch.float32
        )
    )

    print("vary hidden dim", "torch.bfloat16")
    print(
        profiler.profile_varying_hidden_dim(
            16, 512, [256, 512, 768, 1024, 2048], [8, 8, 8, 8, 8], dtype=torch.bfloat16
        )
    )

    print("vary batch size", "torch.float32")
    print(
        profiler.profile_varying_batch_size(
            [16, 16, 24, 32], 512, 512, 8, dtype=torch.float32
        )
    )

    print("vary batch size", "torch.bfloat16")
    print(
        profiler.profile_varying_batch_size(
            [16, 16, 24, 32], 512, 512, 8, dtype=torch.bfloat16
        )
    )

    print("memory", "torch.float32")
    print(profiler.profile_memory_usage(16, 1024, 512, 8, torch.float32))

    print("memory", "torch.bfloat16")
    print(profiler.profile_memory_usage(16, 1024, 512, 8, torch.bfloat16))

    pos_embedder = VideoRopePosition3DEmb(
        head_dim=64, len_h=32, len_w=32, len_t=16, base_fps=30
    ).to("cuda")
    x = torch.randn((5, 16, 32, 32, 64))
    y = apply_rotary_pos_emb(
        torch.randn((5, 8, 50, 64), device="cuda"),
        freqs=pos_embedder.generate_embeddings(
            (5, 16, 32, 32, 64),
        ),
    )
