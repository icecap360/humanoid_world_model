import torch.nn as nn
import torch 
from .common_blocks import SinusoidalPosEmb, VideoPositionEmb, ActionPositionEmb
import torch.functional as F
from einops import pack, unpack
from .dit_video_blocks import PatchVideo, MMDiTBlock, FinalLayer
from .model import Model

class VideoDiTModel(Model):
    def __init__(self, 
                dim_C,
                dim_T_past,
                dim_T_future,
                dim_L_past,
                dim_L_future,
                dim_W,
                dim_h,
                dim_act,
                dim_hidden,
                patch_lw,
                n_layers,
                n_head,
                cfg_prob,
                patch_t=1,
                device='cuda',
        ):
        super().__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.patch_lw = patch_lw
        self.patch_t = patch_t

        self.dim_C, self.dim_Tf, self.dim_Tp, self.dim_H, self.dim_W = dim_C, dim_T_future, dim_T_past, dim_h, dim_W
        self.dim_Lp = dim_L_past
        self.dim_Lf = dim_L_future
        
        self.dim_act = dim_act
        self.dim_hidden = dim_hidden
        self.dim_head = self.dim_hidden // self.n_head
        self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(self.dim_hidden, theta=300 ),
            nn.Linear(self.dim_hidden , self.dim_hidden * 4),
            nn.SiLU(),
            nn.Linear(self.dim_hidden * 4, self.dim_hidden)
        )
        self.action_embedder = nn.Sequential(
            nn.Linear(self.dim_act , self.dim_hidden * 4),
            nn.SiLU(),
            nn.Linear(self.dim_hidden * 4, self.dim_hidden)
        )
        self.patcher_f = PatchVideo(
                dim_c=self.dim_C,
                dim_t=self.dim_Lf,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s = self.patch_lw,
                patch_t = self.patch_t,
                )
        self.patcher_p = PatchVideo(
                dim_c=self.dim_C,
                dim_t=self.dim_Lp,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s = self.patch_lw,
                patch_t = self.patch_t,
                )
        self.action_pos_embed = ActionPositionEmb(self.dim_Tp + self.dim_Tf, self.dim_head, theta=10000.0) # both future and past tokens simultaneously
        self.video_pos_embed = VideoPositionEmb(
            head_dim=self.dim_head,
            len_h=self.dim_H,
            len_w=self.dim_W,
            len_t=self.dim_Lp + self.dim_Lf, # notice how we embed both future and past tokens simultaneously
            theta=10000.0,
            device=device
        )
        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            block = None
            if i == self.n_layers - 1:
                block = MMDiTBlock(
                    self.dim_hidden,
                    self.dim_hidden,
                    skip_context_ff = True
                )
            else:
                block = MMDiTBlock(
                    self.dim_hidden,
                    self.dim_hidden,
                )
            self.blocks.append(block)
        self.final_layer = FinalLayer(
            self.dim_hidden,
            patch_lw=self.patch_lw,
            patch_t=self.patch_t,
            out_channels=self.dim_C
        )

        self.empty_past_frames_emb = torch.zeros((self.dim_C, self.dim_Lp, self.dim_H, self.dim_W)) # nn.Parameter(torch.zeros((self.dim_C, self.dim_Lp, self.dim_H, self.dim_W)))
        self.empty_past_actions_emb = torch.zeros((self.dim_Tp, self.dim_act)) # nn.Parameter(torch.zeros((self.dim_Tp, self.dim_act)))
        self.empty_future_actions_emb = torch.zeros((self.dim_Tf, self.dim_act)) # nn.Parameter(torch.zeros((self.dim_Tf, self.dim_act)))
        
        self.cfg_prob = cfg_prob
        # self.conditioning_manager = conditioning_manager
        # self.conditioning = conditioning
        self.initialize_weights()

    def context_drop(self, batch, use_cfg, device, force_drop_ids=False):
        """ USING TORCH.CONTEXT
        Drops labels to enable classifier-free guidance.
        """
        b, c, tp, h, w = batch['noisy_latents'].shape
        if force_drop_ids == False and use_cfg:
            drop_ids = torch.rand(b, device=device) < self.cfg_prob
            batch['past_latents'][drop_ids, :] = self.empty_past_frames_emb.to(device)
            batch['past_actions'][drop_ids, :] = self.empty_past_actions_emb.to(device)
            batch['future_actions'][drop_ids, :] = self.empty_future_actions_emb.to(device)
        elif force_drop_ids == False and use_cfg == False:
            pass
        elif force_drop_ids == True:            
            # batch['past_latents'] = self.empty_past_frames_emb.repeat(b,1,1,h,w).to(device)
            # batch['past_actions'] = self.empty_past_actions_emb.repeat(b,tp,1).to(device)
            # b, _, tf, _, _ = batch['noisy_latents'].shape
            # batch['future_actions'] = self.empty_future_actions_emb.repeat(b,tf,1).to(device)

            batch['past_latents'] = self.empty_past_frames_emb.repeat(b,1,1,1,1).to(device)
            batch['past_actions'] = self.empty_past_actions_emb.repeat(b,1,1).to(device)
            b, _, tf, _, _ = batch['noisy_latents'].shape
            batch['future_actions'] = self.empty_future_actions_emb.repeat(b,1,1).to(device)
        return batch
    
    def forward(self, batch, time, device='cuda:1', use_cfg=False):
        device = batch['noisy_latents'].device

        force_drop_ids = batch.get('past_latents') is None
        batch = self.context_drop(batch, use_cfg, device, force_drop_ids=force_drop_ids)
        
        fv = batch['noisy_latents']
        pv = batch['past_latents']
        pa = batch['past_actions']
        fa = batch['future_actions']


        pa = self.action_embedder(pa)
        fa = self.action_embedder(fa)
        fv = self.patcher_f(fv)
        pv = self.patcher_p(pv)
        time = self.time_embedder(time)
        
        for i in range(self.n_layers):
            fv, pv, fa, pa = self.blocks[i](fv, pv, fa, pa, time, self.video_pos_embed, self.action_pos_embed)
        
        fv = self.final_layer(fv, time)
        fv = self.patcher_f.unpatchify(fv)
        return fv       
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.action_embedder.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder[1].weight, std=0.02)
        nn.init.normal_(self.time_embedder[3].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
if __name__ == '__main__':
    t = 8
    c = 64
    model = VideoDiTModel(c, t, t, 32, 32, 26, 128, 2, 6, 8, 0.2, device='cpu')
    model = model.to('cpu')
    fv = torch.randn((5, c, t, 32, 32), device='cpu')
    pv = torch.randn((5, c, t, 32, 32), device='cpu')
    fa = torch.randn((5, t, 26), device='cpu')
    pa = torch.randn((5, t, 26), device='cpu')
    t = torch.randint(0, 100, (5, 1), device='cpu')
    fv = model({
        'noisy_future_frames':fv, 
        'past_frames':pv, 
        'future_actions':fa, 
        'past_actions':pa, 
    }, t)