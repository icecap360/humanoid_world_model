import torch.nn as nn
import torch 
from .common_blocks import SinusoidalPosEmb, Attention, QKV, JointAttention, JointFactorizedAttention, FeedForward
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
        self.proj = nn.Conv3d(dim_c,
                              dim_hidden, 
                              kernel_size=block_size,
                              stride=block_size)

    def forward(self, x):
        b, c, t, h, w = x.shape
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
    b, c, t, h, w = x.shape
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
        b, c, t, h, w = x.shape
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
        _, _, _, h, w = fv.shape
        
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
        
        fv = fv + gate(self.attn_norm_cv(fv_res), fv_post_attn_gamma) 
        pv = pv + gate(self.attn_norm_pv(pv_res), pv_post_attn_gamma) 
        fa = fa + gate(self.attn_norm_ca(fa_res), fa_post_attn_gamma) 
        pa = pa + gate(self.attn_norm_pa(pa_res), pa_post_attn_gamma) 

        fv_res = modulate(self.ff_norm_cv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_cv(fv_res) 
        fv = fv + gate(self.ff_norm_cv(fv_res), fv_post_ff_gamma) 
        fv = rearrange(fv, 'b (t h w) d -> b d t h w', h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta) 
            fa_res = modulate(self.ff_norm_ca(fa), fa_pre_ff_gamma, fa_pre_ff_beta) 
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta) 

            pv_res = self.ff_pv(pv_res) 
            fa_res = self.ff_ca(fa_res) 
            pa_res = self.ff_pa(pa_res) 
        
            pv = pv + gate(self.ff_norm_pv(pv_res), pv_post_ff_gamma) 
            fa = fa + gate(self.ff_norm_ca(fa_res), fa_post_ff_gamma) 
            pa = pa + gate(self.ff_norm_pa(pa_res), pa_post_ff_gamma) 
        
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
        
        self.time_scale_shift = AdaLayerNormZero(time_dim, token_dim, param_factor=6, n_context=3)
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        
        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)
        
        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pv = self.qkv_fv
        self.qkv_fa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pa = self.qkv_fa
        
        self.joint_attn = JointAttention(
            token_dim, 
            num_heads=self.num_heads,
        )
        
        self.ff_cv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            self.ff_pv = self.ff_cv
            self.ff_ca = FeedForward(token_dim, act=self.act)
            self.ff_pa = self.ff_ca
        
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()
        
class MMDiTBlockFullSharing(MMDiTBlock):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        nn.Module.__init__(self)
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = False
        
        self.time_scale_shift = AdaLayerNormZero(time_dim, token_dim, param_factor=6, n_context=3)
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        
        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)
        
        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pv = self.qkv_fv
        self.qkv_fa = self.qkv_fv
        self.qkv_pa = self.qkv_fv
        
        self.joint_attn = JointAttention(
            token_dim, 
            num_heads=self.num_heads,
        )
        
        self.ff_cv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            self.ff_pv = self.ff_cv
            self.ff_ca = self.ff_cv
            self.ff_pa = self.ff_cv
        
        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()

class FinalLayer(nn.Module):
    """
    Based off of Facebook's DiT
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, patch_lw, patch_t, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = AdaLayerNormZero(hidden_size, hidden_size, 2.0) # nn.Sequential( nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.linear = nn.Linear(hidden_size, patch_lw * patch_lw * patch_t * out_channels, bias=True)
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