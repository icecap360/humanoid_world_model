import torch.nn as nn
import torch 
from .common_blocks import SinusoidalPosEmb, Attention, QKV, Q, KV, JointAttention, JointFactorizedAttention, FeedForward
import torch.functional as F
from einops import rearrange, pack, unpack

# notice how there is both a modulate and a gat
def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
def gate(x, scale):
    return x * scale.unsqueeze(1)

class PatchVideo(nn.Module):
    def __init__(self, 
                 dim_c,
                 dim_t,
                 dim_h,
                 dim_w,
                 dim_hidden,
                 patch_s = 2,
                 patch_t = 1
                 ):
        super().__init__()
        self.patch_s = patch_s
        self.patch_t = patch_t
        self.dim_c = dim_c
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_hidden = dim_hidden
        block_size = (self.patch_t, self.patch_s, self.patch_s)
        self.proj = nn.Sequential(
            # nn.GroupNorm(8, 16),
            # nn.GELU(),
            nn.Conv3d(dim_c,
                              dim_hidden, 
                              kernel_size=block_size,
                              stride=block_size),
            )

    def forward(self, x):
        x = self.proj(x)
        return x
    def unpatchify(self, x):
        """
        x: (N, T, L, W, patch_length * patch_width * C)
        imgs: (N, C, D, H, W)
        """
        dim_chnl = self.dim_c
        pl = self.patch_s
        pw = self.patch_s
        pt = self.patch_t

        dim_time = self.dim_t
        dim_length = self.dim_h
        dim_width = self.dim_w

        b, d, t, h, w = x.shape
        x = rearrange(x, 'b d t h w -> b (t h w) d', 
                      b=b, d=d, t=t, h=h, w=w)
        
        # Reshape to (N, num_time_patches, num_length_patches, num_width_patches, patch_length, patch_width, C)
        x = x.reshape(shape=(b, t, h, w, pl, pw, dim_chnl))

        # Transpose to (N, C, num_time_patches, patch_length, num_length_patches, patch_width, num_width_patches)
        x = torch.einsum('ntlwpqc->nctlpwq', x)

        # Reshape to (N, C, num_time_patches * patch_length, num_length_patches * patch_width, num_width_patches)
        imgs = x.reshape(shape=(b, dim_chnl, t * pt, h  * pl, w* pw))
        return imgs

def interleave_masks_2d(x, binary_vector):
    binary_vector = binary_vector.to(torch.int32)
    binary_mask = torch.zeros((b, t, h, w), device=x.device, dtype=x.dtype)
    binary_mask[torch.where(binary_vector)] = 1.0
    binary_mask = binary_mask.unsqueeze(1)
    x = torch.concatenate((x, binary_mask), 1)
    return x
def interleave_masks_1d(x, binary_vector):
    b, t, c = x.shape
    binary_vector = binary_vector.to(torch.int32)
    binary_mask = torch.zeros((b, t), device=x.device, dtype=x.dtype)
    binary_mask[torch.where(binary_vector)] = 1.0
    binary_mask = binary_mask.unsqueeze(-1)
    x = torch.concatenate((x, binary_mask), -1)
    return x

def interleave_actions(x, action):
    b, c1, t, h, w = x.shape
    b, c2, t = action.shape
    action_expanded = action.view(b, c2, t, 1, 1).expand(-1, -1, -1, h, w)
    # Concatenate along the channel dimension
    x = torch.cat([x, action_expanded], dim=1)  # New shape: (B, C + C_action, T, H, W)
    return x

class PatchVideoTempMask(PatchVideo):
    def __init__(self, 
                 dim_c,
                 dim_t,
                 dim_h,
                 dim_w,
                 dim_hidden,
                 patch_s = 2,
                 patch_t = 1
                 ):
        nn.Module.__init__(self)
        self.patch_s = patch_s
        self.patch_t = patch_t
        self.dim_c = dim_c
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_hidden = dim_hidden
        block_size = (self.patch_t, self.patch_s, self.patch_s)
        self.proj = nn.Conv3d(dim_c + 1,
                              dim_hidden, 
                              kernel_size=block_size,
                              stride=block_size)
    def forward(self, x):
        x = self.proj(x)
        return x
    
class AdaLayerNormZero(nn.Module):
    def __init__(self, input_dim, embedding_dim: int, param_factor, bias=True, n_context=0):
        super().__init__()
        self.silu = nn.SiLU()
        self.n_context = int(n_context)
        self.n_chunks = (1 + self.n_context) * int(param_factor)
        self.linear = nn.Linear(input_dim, self.n_chunks * embedding_dim, bias=bias)
        self.initialize_weights()

    def forward(self, x):
        emb = self.linear(self.silu(x))
        ret = emb.chunk(self.n_chunks , dim=1)
        return ret # [x.unsqueeze(1) for x in ret]
    def initialize_weights(self):
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)
class AdaLayerNorm(AdaLayerNormZero):
    def initialize_weights(self):
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, std=0.02)
class AdaLayerNormIdentity(AdaLayerNormZero):
    def initialize_weights(self):
        nn.init.constant_(self.linear.weight, 1.0)
        nn.init.constant_(self.linear.bias, 0.0)
        
class MMDiTBlock(nn.Module):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        super().__init__()
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = True
        
        self.time_scale_shift = AdaLayerNormZero(time_dim, token_dim, param_factor=6, n_context=3)
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        
        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)
        
        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_fa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        
        self.joint_attn = JointAttention(
            token_dim, 
            num_heads=self.num_heads,
        )
        
        self.ff_cv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            self.ff_pv = FeedForward(token_dim, act=self.act)
            self.ff_ca = FeedForward(token_dim, act=self.act)
            self.ff_pa = FeedForward(token_dim, act=self.act)
        
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()
    
    def forward(self, fv, pv, fa, pa, timesteps, video_pos_embed, action_pos_embed):
        '''
        fv - future video
        pv - past video
        fa - future actions 
        pa - past actions
        '''
        h, w = fv.shape[-2], fv.shape[-1]
        
        (
            fv_pre_attn_gamma,
            fv_post_attn_gamma,
            fv_pre_ff_gamma,
            fv_post_ff_gamma,
            fv_pre_attn_beta,
            fv_pre_ff_beta,
            
            pv_pre_attn_gamma,
            pv_post_attn_gamma,
            pv_pre_ff_gamma,
            pv_post_ff_gamma,
            pv_pre_attn_beta,
            pv_pre_ff_beta,
            
            fa_pre_attn_gamma,
            fa_post_attn_gamma,
            fa_pre_ff_gamma,
            fa_post_ff_gamma,
            fa_pre_attn_beta,
            fa_pre_ff_beta,

            pa_pre_attn_gamma,
            pa_post_attn_gamma,
            pa_pre_ff_gamma,
            pa_post_ff_gamma,
            pa_pre_attn_beta,
            pa_pre_ff_beta
        ) = self.time_scale_shift(timesteps)

        fv = rearrange(fv, 'b d t h w -> b (t h w) d')
        pv = rearrange(pv, 'b d t h w -> b (t h w) d')
        
        fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta) 
        pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta) 
        fa_res = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta) 
        pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta) 
        
        q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
        q_pv, k_pv, v_pv = self.qkv_pv(pv_res)
        q_fa, k_fa, v_fa = self.qkv_fa(fa_res)
        q_pa, k_pa, v_pa = self.qkv_pa(pa_res)
        
        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

        q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

        fv_res, pv_res, fa_res, pa_res = self.joint_attn([
            (q_fv, k_fv, v_fv),
            (q_pv, k_pv, v_pv),
            (q_fa, k_fa, v_fa),
            (q_pa, k_pa, v_pa),
        ])
        
        fv = fv + gate(fv_res, fv_post_attn_gamma) 
        pv = pv + gate(pv_res, pv_post_attn_gamma) 
        fa = fa + gate(fa_res, fa_post_attn_gamma) 
        pa = pa + gate(pa_res, pa_post_attn_gamma) 

        fv_res = modulate(self.ff_norm_cv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_cv(fv_res) 
        fv = fv + gate(fv_res, fv_post_ff_gamma) 
        fv = rearrange(fv, 'b (t h w) d -> b d t h w', h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta) 
            fa_res = modulate(self.ff_norm_ca(fa), fa_pre_ff_gamma, fa_pre_ff_beta) 
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta) 

            pv_res = self.ff_pv(pv_res) 
            fa_res = self.ff_ca(fa_res) 
            pa_res = self.ff_pa(pa_res) 
        
            pv = pv + gate(pv_res, pv_post_ff_gamma) 
            fa = fa + gate(fa_res, fa_post_ff_gamma) 
            pa = pa + gate(pa_res, pa_post_ff_gamma) 
        
        pv = rearrange(pv, 'b (t h w) d -> b d t h w', h=h, w=w)
        return fv, pv, fa, pa

    def pos_embed_pf(self, p, f, pos_embedder):
        a, pack_info = pack([p, f], 'b h * d')
        a = pos_embedder(a)
        p, f = unpack(a, pack_info, 'b h * d')
        return p, f
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.qkv_fv.apply(_basic_init)
        self.qkv_pv.apply(_basic_init)
        self.qkv_fa.apply(_basic_init)
        self.qkv_pa.apply(_basic_init)
        if not self.skip_context_ff:
            self.ff_cv.apply(_basic_init)
            self.ff_pv.apply(_basic_init)
            self.ff_ca.apply(_basic_init)
            self.ff_pa.apply(_basic_init)

class MMDiTBlockModalitySharing(MMDiTBlock):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        nn.Module.__init__(self)
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = False

        self.time_scale_shift_v = AdaLayerNormZero(
            time_dim, token_dim, param_factor=6, n_context=0
        )
        self.time_scale_shift_a = AdaLayerNormZero(
            time_dim, token_dim, param_factor=6, n_context=0
        )
        # notice how elementwise_affine=False because we are using AdaLNZero blocks

        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_fa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)

        self.joint_attn = JointAttention(
            token_dim,
            num_heads=self.num_heads,
        )

        self.ff_fv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            self.ff_fa = FeedForward(token_dim, act=self.act)

        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_fv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_fa = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()

    def forward(self, fv, pv, fa, pa, timesteps, video_pos_embed, action_pos_embed):
        """
        fv - future video
        pv - past video
        fa - future actions
        pa - past actions
        """
        h, w = fv.shape[-2], fv.shape[-1]

        (
            fv_pre_attn_gamma,
            fv_post_attn_gamma,
            fv_pre_ff_gamma,
            fv_post_ff_gamma,
            fv_pre_attn_beta,
            fv_pre_ff_beta,
        ) = self.time_scale_shift_v(timesteps)

        pv_pre_attn_gamma = fv_pre_attn_gamma
        pv_post_attn_gamma = fv_post_attn_gamma
        pv_pre_ff_gamma = fv_pre_ff_gamma
        pv_post_ff_gamma = fv_post_ff_gamma
        pv_pre_attn_beta = fv_pre_attn_beta
        pv_pre_ff_beta = fv_pre_ff_beta

        (
            fa_pre_attn_gamma,
            fa_post_attn_gamma,
            fa_pre_ff_gamma,
            fa_post_ff_gamma,
            fa_pre_attn_beta,
            fa_pre_ff_beta,
        ) = self.time_scale_shift_a(timesteps)

        pa_pre_attn_gamma = fa_pre_attn_gamma
        pa_post_attn_gamma = fa_post_attn_gamma
        pa_pre_ff_gamma = fa_pre_ff_gamma
        pa_post_ff_gamma = fa_post_ff_gamma
        pa_pre_attn_beta = fa_pre_attn_beta
        pa_pre_ff_beta = fa_pre_ff_beta

        fv = rearrange(fv, "b d t h w -> b (t h w) d")
        pv = rearrange(pv, "b d t h w -> b (t h w) d")

        fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta)
        pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta)
        fa_res = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta)
        pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta)

        q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
        q_pv, k_pv, v_pv = self.qkv_fv(pv_res)
        q_fa, k_fa, v_fa = self.qkv_fa(fa_res)
        q_pa, k_pa, v_pa = self.qkv_fa(pa_res)

        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

        q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

        fv_res, pv_res, fa_res, pa_res = self.joint_attn(
            [
                (q_fv, k_fv, v_fv),
                (q_pv, k_pv, v_pv),
                (q_fa, k_fa, v_fa),
                (q_pa, k_pa, v_pa),
            ]
        )

        fv = fv + gate(fv_res, fv_post_attn_gamma)
        pv = pv + gate(pv_res, pv_post_attn_gamma)
        fa = fa + gate(fa_res, fa_post_attn_gamma)
        pa = pa + gate(pa_res, pa_post_attn_gamma)

        fv_res = modulate(self.ff_norm_fv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_fv(fv_res)
        fv = fv + gate(fv_res, fv_post_ff_gamma)
        fv = rearrange(fv, "b (t h w) d -> b d t h w", h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta)
            fa_res = modulate(self.ff_norm_fa(fa), fa_pre_ff_gamma, fa_pre_ff_beta)
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta)

            pv_res = self.ff_fv(pv_res)
            fa_res = self.ff_fa(fa_res)
            pa_res = self.ff_fa(pa_res)

            pv = pv + gate(pv_res, pv_post_ff_gamma)
            fa = fa + gate(fa_res, fa_post_ff_gamma)
            pa = pa + gate(pa_res, pa_post_ff_gamma)

        pv = rearrange(pv, "b (t h w) d -> b d t h w", h=h, w=w)
        return fv, pv, fa, pa

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.qkv_fv.apply(_basic_init)
        self.qkv_fa.apply(_basic_init)
        if not self.skip_context_ff:
            self.ff_fv.apply(_basic_init)
            self.ff_fa.apply(_basic_init)


class MMDiTBlockFullSharing(MMDiTBlock):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        nn.Module.__init__(self)
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = False

        self.time_scale_shift = AdaLayerNormZero(
            time_dim, token_dim, param_factor=6, n_context=0
        )
        # notice how elementwise_affine=False because we are using AdaLNZero blocks

        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)

        self.joint_attn = JointAttention(
            token_dim,
            num_heads=self.num_heads,
        )

        self.ff_fv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff

        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_fv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()

    def forward(self, fv, pv, fa, pa, timesteps, video_pos_embed, action_pos_embed):
        """
        fv - future video
        pv - past video
        fa - future actions
        pa - past actions
        """
        h, w = fv.shape[-2], fv.shape[-1]

        (
            fv_pre_attn_gamma,
            fv_post_attn_gamma,
            fv_pre_ff_gamma,
            fv_post_ff_gamma,
            fv_pre_attn_beta,
            fv_pre_ff_beta,
        ) = self.time_scale_shift(timesteps)

        pv_pre_attn_gamma = fv_pre_attn_gamma
        pv_post_attn_gamma = fv_post_attn_gamma
        pv_pre_ff_gamma = fv_pre_ff_gamma
        pv_post_ff_gamma = fv_post_ff_gamma
        pv_pre_attn_beta = fv_pre_attn_beta
        pv_pre_ff_beta = fv_pre_ff_beta

        fa_pre_attn_gamma = fv_pre_attn_gamma
        fa_post_attn_gamma = fv_post_attn_gamma
        fa_pre_ff_gamma = fv_pre_ff_gamma
        fa_post_ff_gamma = fv_post_ff_gamma
        fa_pre_attn_beta = fv_pre_attn_beta
        fa_pre_ff_beta = fv_pre_ff_beta

        pa_pre_attn_gamma = fa_pre_attn_gamma
        pa_post_attn_gamma = fa_post_attn_gamma
        pa_pre_ff_gamma = fa_pre_ff_gamma
        pa_post_ff_gamma = fa_post_ff_gamma
        pa_pre_attn_beta = fa_pre_attn_beta
        pa_pre_ff_beta = fa_pre_ff_beta

        fv = rearrange(fv, "b d t h w -> b (t h w) d")
        pv = rearrange(pv, "b d t h w -> b (t h w) d")

        fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta)
        pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta)
        fa_res = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta)
        pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta)

        q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
        q_pv, k_pv, v_pv = self.qkv_fv(pv_res)
        q_fa, k_fa, v_fa = self.qkv_fv(fa_res)
        q_pa, k_pa, v_pa = self.qkv_fv(pa_res)

        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

        q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

        fv_res, pv_res, fa_res, pa_res = self.joint_attn(
            [
                (q_fv, k_fv, v_fv),
                (q_pv, k_pv, v_pv),
                (q_fa, k_fa, v_fa),
                (q_pa, k_pa, v_pa),
            ]
        )

        fv = fv + gate(fv_res, fv_post_attn_gamma)
        pv = pv + gate(pv_res, pv_post_attn_gamma)
        fa = fa + gate(fa_res, fa_post_attn_gamma)
        pa = pa + gate(pa_res, pa_post_attn_gamma)

        fv_res = modulate(self.ff_norm_fv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_fv(fv_res)
        fv = fv + gate(fv_res, fv_post_ff_gamma)
        fv = rearrange(fv, "b (t h w) d -> b d t h w", h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta)
            fa_res = modulate(self.ff_norm_ca(fa), fa_pre_ff_gamma, fa_pre_ff_beta)
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta)

            pv_res = self.ff_fv(pv_res)
            fa_res = self.ff_fv(fa_res)
            pa_res = self.ff_fv(pa_res)

            pv = pv + gate(pv_res, pv_post_ff_gamma)
            fa = fa + gate(fa_res, fa_post_ff_gamma)
            pa = pa + gate(pa_res, pa_post_ff_gamma)

        pv = rearrange(pv, "b (t h w) d -> b d t h w", h=h, w=w)
        return fv, pv, fa, pa

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.qkv_fv.apply(_basic_init)
        if not self.skip_context_ff:
            self.ff_fv.apply(_basic_init)



class MMDiTSplitAttentionBlock(MMDiTBlock):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        nn.Module.__init__(self)
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = True
        
        self.time_scale_shift_v = AdaLayerNormZero(time_dim, token_dim, param_factor=6, n_context=0)
        self.time_scale_shift_a = AdaLayerNormZero(time_dim, token_dim, param_factor=6, n_context=0)
        self.time_scale_shift_cross = AdaLayerNormZero(time_dim, token_dim, param_factor=3, n_context=0)
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        
        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)
        
        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        # self.qkv_pv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_fa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        # self.qkv_pa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        
        self.crossattn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.crossattn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.crossattn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.crossattn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)
        
        self.q_fv = Q(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.kv_pv = KV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.kv_fa = KV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.kv_pa = KV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        
        self.joint_attns = nn.ModuleList(
            [JointAttention(
                token_dim, 
                num_heads=self.num_heads,
                ) for _ in range(2)])
        self.cross_attn = JointAttention(
                token_dim, 
                num_heads=self.num_heads,
                )
        
        self.ff_fv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            # self.ff_pv = FeedForward(token_dim, act=self.act)
            self.ff_fa = FeedForward(token_dim, act=self.act)
            # self.ff_pa = FeedForward(token_dim, act=self.act)
        
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_fv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_fa = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()
    
    def forward(self, fv, pv, fa, pa, timesteps, video_pos_embed, action_pos_embed):
        '''
        fv - future video
        pv - past video
        fa - future actions 
        pa - past actions
        '''
        h, w = fv.shape[-2], fv.shape[-1]
        
        (
            fv_pre_attn_gamma,
            fv_post_attn_gamma,
            fv_pre_ff_gamma,
            fv_post_ff_gamma,
            fv_pre_attn_beta,
            fv_pre_ff_beta,
        ) = self.time_scale_shift_v(timesteps)

        pv_pre_attn_gamma = fv_pre_attn_gamma
        pv_post_attn_gamma = fv_post_attn_gamma
        pv_pre_ff_gamma = fv_pre_ff_gamma
        pv_post_ff_gamma = fv_post_ff_gamma
        pv_pre_attn_beta = fv_pre_attn_beta
        pv_pre_ff_beta = fv_pre_ff_beta

        (
            fa_pre_attn_gamma,
            fa_post_attn_gamma,
            fa_pre_ff_gamma,
            fa_post_ff_gamma,
            fa_pre_attn_beta,
            fa_pre_ff_beta,
        ) = self.time_scale_shift_a(timesteps)

        pa_pre_attn_gamma = fa_pre_attn_gamma
        pa_post_attn_gamma = fa_post_attn_gamma
        pa_pre_ff_gamma = fa_pre_ff_gamma
        pa_post_ff_gamma = fa_post_ff_gamma
        pa_pre_attn_beta = fa_pre_attn_beta
        pa_pre_ff_beta = fa_pre_ff_beta

        (   fv_pre_crossattn_gamma,
            fv_post_crossattn_gamma,
            fv_pre_crossattn_beta) = self.time_scale_shift_cross(timesteps)
        fv = rearrange(fv, 'b d t h w -> b (t h w) d')
        pv = rearrange(pv, 'b d t h w -> b (t h w) d')
        
        fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta) 
        pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta) 
        fa_res = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta) 
        pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta) 
        
        q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
        q_pv, k_pv, v_pv = self.qkv_fv(pv_res)
        q_fa, k_fa, v_fa = self.qkv_fa(fa_res)
        q_pa, k_pa, v_pa = self.qkv_fa(pa_res)
        
        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

        q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

        fv_res = self.joint_attns[0]([(q_fv, k_fv, v_fv)])[0]
        pv_res = self.joint_attns[0]([(q_pv, k_pv, v_pv)])[0]
        fa_res = self.joint_attns[1]([(q_fa, k_fa, v_fa)])[0]
        pa_res = self.joint_attns[1]([(q_pa, k_pa, v_pa)])[0]
        
        fv = fv + gate(fv_res, fv_post_attn_gamma) 
        pv = pv + gate(pv_res, pv_post_attn_gamma) 
        fa = fa + gate(fa_res, fa_post_attn_gamma) 
        pa = pa + gate(pa_res, pa_post_attn_gamma) 
        
        ##### Cross attention branch
        fv_res = modulate(self.crossattn_norm_cv(fv), fv_pre_crossattn_gamma, fv_pre_crossattn_beta) 
        pv_res = self.crossattn_norm_pv(pv)
        fa_res = self.crossattn_norm_ca(fa) 
        pa_res = self.crossattn_norm_pa(pa)
        
        q_fv = self.q_fv(fv_res)
        k_pv, v_pv = self.kv_pv(pv_res)
        k_fa, v_fa = self.kv_fa(fa_res)
        k_pa, v_pa = self.kv_pa(pa_res)
        
        q_fv = video_pos_embed(q_fv)
        k_pv = video_pos_embed(k_pv)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)
        
        k = torch.cat((k_pv, k_fa, k_pa), 2)
        v = torch.cat((v_pv, v_fa, v_pa), 2)
        fv_res = self.cross_attn([(q_fv, k, v)])[0]
        
        fv = fv + gate(fv_res, fv_post_crossattn_gamma) 
        #####

        fv_res = modulate(self.ff_norm_fv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_fv(fv_res) 
        fv = fv + gate(fv_res, fv_post_ff_gamma) 
        fv = rearrange(fv, 'b (t h w) d -> b d t h w', h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta) 
            fa_res = modulate(self.ff_norm_fa(fa), fa_pre_ff_gamma, fa_pre_ff_beta) 
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta) 

            pv_res = self.ff_fv(pv_res) 
            fa_res = self.ff_fa(fa_res) 
            pa_res = self.ff_fa(pa_res) 
        
            pv = pv + gate(pv_res, pv_post_ff_gamma) 
            fa = fa + gate(fa_res, fa_post_ff_gamma) 
            pa = pa + gate(pa_res, pa_post_ff_gamma) 
        
        pv = rearrange(pv, 'b (t h w) d -> b d t h w', h=h, w=w)
        return fv, pv, fa, pa

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.qkv_fv.apply(_basic_init)
        # self.qkv_pv.apply(_basic_init)
        self.qkv_fa.apply(_basic_init)
        # self.qkv_pa.apply(_basic_init)
        if not self.skip_context_ff:
            self.ff_fv.apply(_basic_init)
            # self.ff_pv.apply(_basic_init)
            self.ff_fa.apply(_basic_init)
            # self.ff_pa.apply(_basic_init)

class FinalLayer(nn.Module):
    """
    Based off of Facebook's DiT
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, patch_lw, patch_t, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = AdaLayerNormZero(hidden_size, hidden_size, 2.0) # nn.Sequential( nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.linear = nn.Linear(hidden_size, patch_lw * patch_lw * patch_t * out_channels, bias=False)
    def forward(self, x, timesteps):
        b, d, t, h, w = x.shape
        x = rearrange(x, 'b d t h w -> b (t h w) d', 
                      b=b, d=d, t=t, h=h, w=w)
        shift, scale = self.adaLN_modulation(timesteps)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        x = rearrange(x, 'b (t h w) d -> b d t h w', 
                b=b, t=t, h=h, w=w)
        return x

from hyper_connections import HyperConnections, get_init_and_expand_reduce_stream_functions

class MMDiTBlockHyperConnections(MMDiTBlock):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        super().__init__(token_dim, time_dim, num_heads, skip_context_ff)
        init_hyper_conn_fv_attn, expand_stream_fv_attn, reduce_stream_fv_attn = get_init_and_expand_reduce_stream_functions(1)
        init_hyper_conn_pv_attn, expand_stream_pv_attn, reduce_stream_pv_attn = get_init_and_expand_reduce_stream_functions(1)
        init_hyper_conn_fa_attn, expand_stream_fa_attn, reduce_stream_fa_attn = get_init_and_expand_reduce_stream_functions(1)
        init_hyper_conn_pa_attn, expand_stream_pa_attn, reduce_stream_pa_attn = get_init_and_expand_reduce_stream_functions(1)

        # For feed-forward branch on each stream
        init_hyper_conn_fv_ff, expand_stream_fv_ff, reduce_stream_fv_ff = get_init_and_expand_reduce_stream_functions(1)
        init_hyper_conn_pv_ff, expand_stream_pv_ff, reduce_stream_pv_ff = get_init_and_expand_reduce_stream_functions(1)
        init_hyper_conn_fa_ff, expand_stream_fa_ff, reduce_stream_fa_ff = get_init_and_expand_reduce_stream_functions(1)
        init_hyper_conn_pa_ff, expand_stream_pa_ff, reduce_stream_pa_ff = get_init_and_expand_reduce_stream_functions(1)
        # Initialize independent hyperconnection modules for each stream for attention branch
        self.attn_hyper_fv = init_hyper_conn_fv_attn(dim=token_dim)
        self.attn_hyper_pv = init_hyper_conn_pv_attn(dim=token_dim)
        self.attn_hyper_fa = init_hyper_conn_fa_attn(dim=token_dim)
        self.attn_hyper_pa = init_hyper_conn_pa_attn(dim=token_dim)
        
        # and for feed-forward branch
        self.ff_hyper_fv = init_hyper_conn_fv_ff(dim=token_dim)
        self.ff_hyper_pv = init_hyper_conn_pv_ff(dim=token_dim)
        self.ff_hyper_fa = init_hyper_conn_fa_ff(dim=token_dim)
        self.ff_hyper_pa = init_hyper_conn_pa_ff(dim=token_dim)

        self.expand_stream_fv_attn = expand_stream_fv_attn
        self.reduce_stream_fv_attn = reduce_stream_fv_attn
        self.expand_stream_pv_attn = expand_stream_pv_attn
        self.reduce_stream_pv_attn = reduce_stream_pv_attn
        self.expand_stream_fa_attn = expand_stream_fa_attn
        self.reduce_stream_fa_attn = reduce_stream_fa_attn
        self.expand_stream_pa_attn = expand_stream_pa_attn
        self.reduce_stream_pa_attn = reduce_stream_pa_attn

        self.expand_stream_fv_ff = expand_stream_fv_ff
        self.reduce_stream_fv_ff = reduce_stream_fv_ff
        self.expand_stream_pv_ff = expand_stream_pv_ff
        self.reduce_stream_pv_ff = reduce_stream_pv_ff
        self.expand_stream_fa_ff = expand_stream_fa_ff
        self.reduce_stream_fa_ff = reduce_stream_fa_ff
        self.expand_stream_pa_ff = expand_stream_pa_ff
        self.reduce_stream_pa_ff = reduce_stream_pa_ff
    
    def forward(self, fv, pv, fa, pa, timesteps, video_pos_embed, action_pos_embed):
        '''
        fv - future video
        pv - past video
        fa - future actions 
        pa - past actions
        '''
        h, w = fv.shape[-2], fv.shape[-1]
        
        # Obtain time-dependent scaling parameters
        (
            fv_pre_attn_gamma,
            fv_post_attn_gamma,
            fv_pre_ff_gamma,
            fv_post_ff_gamma,
            fv_pre_attn_beta,
            fv_pre_ff_beta,
            
            pv_pre_attn_gamma,
            pv_post_attn_gamma,
            pv_pre_ff_gamma,
            pv_post_ff_gamma,
            pv_pre_attn_beta,
            pv_pre_ff_beta,
            
            fa_pre_attn_gamma,
            fa_post_attn_gamma,
            fa_pre_ff_gamma,
            fa_post_ff_gamma,
            fa_pre_attn_beta,
            fa_pre_ff_beta,

            pa_pre_attn_gamma,
            pa_post_attn_gamma,
            pa_pre_ff_gamma,
            pa_post_ff_gamma,
            pa_pre_attn_beta,
            pa_pre_ff_beta
        ) = self.time_scale_shift(timesteps)

        # Rearranging video streams from (b, d, t, h, w) to (b, t*h*w, d)
        fv = rearrange(fv, 'b d t h w -> b (t h w) d')
        pv = rearrange(pv, 'b d t h w -> b (t h w) d')
        
        # ------------------ Attention Branch ------------------
        # Pre-attention modulation for each stream
        fv_mod = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta)
        pv_mod = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta)
        fa_mod = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta)
        pa_mod = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta)
        
        q_fv, k_fv, v_fv = self.qkv_fv(fv_mod)
        q_pv, k_pv, v_pv = self.qkv_pv(pv_mod)
        q_fa, k_fa, v_fa = self.qkv_fa(fa_mod)
        q_pa, k_pa, v_pa = self.qkv_pa(pa_mod)
        
        # Apply positional embeddings
        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)
        q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)
        
        # Joint attention computation across all streams
        fv_res, pv_res, fa_res, pa_res = self.joint_attn([
            (q_fv, k_fv, v_fv),
            (q_pv, k_pv, v_pv),
            (q_fa, k_fa, v_fa),
            (q_pa, k_pa, v_pa),
        ])
        
        # --- Hyperconnection for Attention Residual Updates ---
        # Process each stream independently using its own hyperconnection:
        # Future video (fv)
        fv_attn = gate(fv_res, fv_post_attn_gamma)
        fv_attn = fv_attn.unsqueeze(1)  # shape: (b,1, tokens, token_dim)
        fv_attn = self.expand_stream_fv_attn(fv_attn)
        fv_attn = self.attn_hyper_fv.decorate_branch(lambda x: x)(fv_attn)
        fv_attn = self.reduce_stream_fv_attn(fv_attn).squeeze(1)
        fv = fv + fv_attn

        # Past video (pv)
        pv_attn = gate(pv_res, pv_post_attn_gamma)
        pv_attn = pv_attn.unsqueeze(1)
        pv_attn = self.expand_stream_pv_attn(pv_attn)
        pv_attn = self.attn_hyper_pv.decorate_branch(lambda x: x)(pv_attn)
        pv_attn = self.reduce_stream_pv_attn(pv_attn).squeeze(1)
        pv = pv + pv_attn

        # Future actions (fa)
        fa_attn = gate(fa_res, fa_post_attn_gamma)
        fa_attn = fa_attn.unsqueeze(1)
        fa_attn = self.expand_stream_fa_attn(fa_attn)
        fa_attn = self.attn_hyper_fa.decorate_branch(lambda x: x)(fa_attn)
        fa_attn = self.reduce_stream_fa_attn(fa_attn).squeeze(1)
        fa = fa + fa_attn

        # Past actions (pa)
        pa_attn = gate(pa_res, pa_post_attn_gamma)
        pa_attn = pa_attn.unsqueeze(1)
        pa_attn = self.expand_stream_pa_attn(pa_attn)
        pa_attn = self.attn_hyper_pa.decorate_branch(lambda x: x)(pa_attn)
        pa_attn = self.reduce_stream_pa_attn(pa_attn).squeeze(1)
        pa = pa + pa_attn
        
        # ------------------ Feed-Forward Branch ------------------
        # Future video feed-forward
        fv_ff = modulate(self.ff_norm_cv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_ff = self.ff_cv(fv_ff)
        fv_ff = gate(fv_ff, fv_post_ff_gamma)
        fv_ff = fv_ff.unsqueeze(1)
        fv_ff = self.expand_stream_fv_ff(fv_ff)
        fv_ff = self.ff_hyper_fv.decorate_branch(lambda x: x)(fv_ff)
        fv_ff = self.reduce_stream_fv_ff(fv_ff).squeeze(1)
        fv = fv + fv_ff

        # Past video feed-forward
        pv_ff = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta)
        pv_ff = self.ff_pv(pv_ff)
        pv_ff = gate(pv_ff, pv_post_ff_gamma)
        pv_ff = pv_ff.unsqueeze(1)
        pv_ff = self.expand_stream_pv_ff(pv_ff)
        pv_ff = self.ff_hyper_pv.decorate_branch(lambda x: x)(pv_ff)
        pv_ff = self.reduce_stream_pv_ff(pv_ff).squeeze(1)
        pv = pv + pv_ff

        # Future actions feed-forward
        if not self.skip_context_ff:
            fa_ff = modulate(self.ff_norm_ca(fa), fa_pre_ff_gamma, fa_pre_ff_beta)
            fa_ff = self.ff_ca(fa_ff)
            fa_ff = gate(fa_ff, fa_post_ff_gamma)
        else:
            fa_ff = torch.zeros_like(fv_ff)
        fa_ff = fa_ff.unsqueeze(1)
        fa_ff = self.expand_stream_fa_ff(fa_ff)
        fa_ff = self.ff_hyper_fa.decorate_branch(lambda x: x)(fa_ff)
        fa_ff = self.reduce_stream_fa_ff(fa_ff).squeeze(1)
        fa = fa + fa_ff

        # Past actions feed-forward
        if not self.skip_context_ff:
            pa_ff = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta)
            pa_ff = self.ff_pa(pa_ff)
            pa_ff = gate(pa_ff, pa_post_ff_gamma)
        else:
            pa_ff = torch.zeros_like(fv_ff)
        pa_ff = pa_ff.unsqueeze(1)
        pa_ff = self.expand_stream_pa_ff(pa_ff)
        pa_ff = self.ff_hyper_pa.decorate_branch(lambda x: x)(pa_ff)
        pa_ff = self.reduce_stream_pa_ff(pa_ff).squeeze(1)
        pa = pa + pa_ff
        pv = rearrange(pv, 'b (t h w) d -> b d t h w', h=h, w=w)
        fv = rearrange(fv, 'b (t h w) d -> b d t h w', h=h, w=w)

        return fv, pv, fa, pa

# class MMDiTBlockHyperConnections(MMDiTBlock):
#     def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
#         super().__init__(token_dim, time_dim, num_heads, skip_context_ff)
#         self.residualfn_attn_fv = HyperConnections(1, dim=token_dim)
#         self.residualfn_attn_pv = HyperConnections(1, dim=token_dim)
#         self.residualfn_attn_fa = HyperConnections(1, dim=token_dim)
#         self.residualfn_attn_pa = HyperConnections(1, dim=token_dim)

#         self.residualfn_ff_fv = HyperConnections(1, dim=token_dim)
#         self.residualfn_ff_pv = HyperConnections(1, dim=token_dim)
#         self.residualfn_ff_fa = HyperConnections(1, dim=token_dim)
#         self.residualfn_ff_pa = HyperConnections(1, dim=token_dim)
    
#     def forward(self, fv, pv, fa, pa, timesteps, video_pos_embed, action_pos_embed):
#         '''
#         fv - future video
#         pv - past video
#         fa - future actions 
#         pa - past actions
#         '''
#         h, w = fv.shape[-2], fv.shape[-1]
        
#         (
#             fv_pre_attn_gamma,
#             fv_post_attn_gamma,
#             fv_pre_ff_gamma,
#             fv_post_ff_gamma,
#             fv_pre_attn_beta,
#             fv_pre_ff_beta,
            
#             pv_pre_attn_gamma,
#             pv_post_attn_gamma,
#             pv_pre_ff_gamma,
#             pv_post_ff_gamma,
#             pv_pre_attn_beta,
#             pv_pre_ff_beta,
            
#             fa_pre_attn_gamma,
#             fa_post_attn_gamma,
#             fa_pre_ff_gamma,
#             fa_post_ff_gamma,
#             fa_pre_attn_beta,
#             fa_pre_ff_beta,

#             pa_pre_attn_gamma,
#             pa_post_attn_gamma,
#             pa_pre_ff_gamma,
#             pa_post_ff_gamma,
#             pa_pre_attn_beta,
#             pa_pre_ff_beta
#         ) = self.time_scale_shift(timesteps)

#         fv = rearrange(fv, 'b d t h w -> b (t h w) d')
#         pv = rearrange(pv, 'b d t h w -> b (t h w) d')
        
#         fv, add_fv_residual = self.residualfn_attn_fv(fv)
#         pv, add_pv_residual = self.residualfn_attn_fv(pv)
#         fa, add_fa_residual = self.residualfn_attn_fv(fa)
#         pa, add_pa_residual = self.residualfn_attn_fv(pa)
        
#         fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta) 
#         pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta) 
#         fa_res = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta) 
#         pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta) 
        
#         q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
#         q_pv, k_pv, v_pv = self.qkv_pv(pv_res)
#         q_fa, k_fa, v_fa = self.qkv_fa(fa_res)
#         q_pa, k_pa, v_pa = self.qkv_pa(pa_res)
        
#         q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
#         k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

#         q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
#         k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

#         fv_res, pv_res, fa_res, pa_res = self.joint_attn([
#             (q_fv, k_fv, v_fv),
#             (q_pv, k_pv, v_pv),
#             (q_fa, k_fa, v_fa),
#             (q_pa, k_pa, v_pa),
#         ])
        
#         fv = add_fv_residual(gate(fv_res, fv_post_attn_gamma))
#         pv = add_pv_residual(gate(pv_res, pv_post_attn_gamma))
#         fa = add_fa_residual(gate(fa_res, fa_post_attn_gamma))
#         pa = add_pa_residual(gate(pa_res, pa_post_attn_gamma))

#         fv, add_fv_residual = self.residualfn_ff_fv(fv)
#         pv, add_pv_residual = self.residualfn_ff_fv(pv)
#         fa, add_fa_residual = self.residualfn_ff_fv(fa)
#         pa, add_pa_residual = self.residualfn_ff_fv(pa)

#         fv_res = modulate(self.ff_norm_cv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
#         fv_res = self.ff_cv(fv_res) 
#         fv = add_fv_residual(gate(fv_res, fv_post_ff_gamma) )

#         if not self.skip_context_ff:
#             pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta) 
#             fa_res = modulate(self.ff_norm_ca(fa), fa_pre_ff_gamma, fa_pre_ff_beta) 
#             pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta) 

#             pv_res = self.ff_pv(pv_res) 
#             fa_res = self.ff_ca(fa_res) 
#             pa_res = self.ff_pa(pa_res) 

#             pv = add_pv_residual(gate(pv_res, pv_post_ff_gamma))
#             fa = add_fa_residual(gate(fa_res, fa_post_ff_gamma))
#             pa = add_pa_residual(gate(pa_res, pa_post_ff_gamma))
        
#         fv = rearrange(fv, 'b (t h w) d -> b d t h w', h=h, w=w)
#         pv = rearrange(pv, 'b (t h w) d -> b d t h w', h=h, w=w)
#         return fv, pv, fa, pa


if __name__ == '__main__':
    x = torch.randn((5, 16, 8, 64, 64))
    binary_vector = torch.ones((5, 8))
    actions = torch.ones((5, 16, 8))
    x = interleave_actions(x, actions)
    binary_vector[1, 0:3] = 0
    x = interleave_masks_2d(x, binary_vector)
    print(x.shape)
    patcher= PatchVideo(patch_t=2, in_chans=16+16+1)
    x = patcher(x)
    print(x.shape)
    
    x = torch.randn((5, 4096, 768))
    y = torch.randn((5, 4096, 768))
    a = torch.randn((5, 1, 768))
    qkv = QKV(768)
    q_x, k_x, v_x = qkv(x)
    q_y, k_y, v_y = qkv(y)
    q_a, k_a, v_a = qkv(a)
    
    joint_attn = JointAttention(768, 8)
    
    # new_x, new_y, new_a = joint_attn([
    #     [q_x, k_x, v_x],
    #     [q_y, k_y, v_y],
    #     [q_a, k_a, v_a]
    # ])
    
    x = torch.randn((5, 64, 64, 768))
    y = torch.randn((5, 64, 64, 768))
    joint_attn = JointFactorizedAttention(768, 8)
    new_x, new_y = joint_attn([
        qkv(x),
        qkv(y),
    ])
    
    t_embedder = nn.Sequential(
            SinusoidalPosEmb(768 ),
            nn.Linear(768 , 768 * 4),
            nn.SiLU(),
            nn.Linear(768 * 4, 768)
        )
    timesteps = t_embedder(torch.tensor([[5],[6],[7],[8],[9]], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1))