import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from .unet_blocks import UNetDownBlock, UNetDownBlockCrossAttn, UNetMidBlock, UNetUpBlock, UNetUpBlockCrossAttn, SinusoidalPosEmb
from .tokenizers import TextEmbedder

class UNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 block_out_channels, 
                 time_embed_dim,
                 cfg_prob=0,
                 text_tokenizer: TextEmbedder = None,
                 conditioning = 'text',
                 attention_resolutions=[],
                 conv_act=nn.SiLU(),
                 down_block_types=None, 
                 up_block_types=None):
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.conv_act = conv_act
        self.block_out_channels = block_out_channels 
        self.down_block_types = down_block_types 
        self.up_block_types = up_block_types 
        self.n_layers = len(block_out_channels)
        
        self.time_dim = time_embed_dim
        self.time_hidden_dim = time_embed_dim*4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim ),
            nn.Linear(self.time_dim , self.time_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.time_hidden_dim, self.time_hidden_dim)
        )

        self.conv_in = nn.Sequential(
            # self.conv_act, 
            # nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)))
        self.conditioning = conditioning
        self.cfg_prob = cfg_prob
        if self.conditioning == 'text':
            self.context_dim = 512
            self.empty_context = nn.Parameter(torch.zeros(self.context_dim))
            self.text_tokenizer = text_tokenizer
        else:
            self.context_dim = None

        self.attention_resolutions = attention_resolutions

        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        ds = 1
        for i in range(self.n_layers):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            add_downsample = True
            if i == self.n_layers -1:
                add_downsample = False
            if ds in attention_resolutions:
                self.encoder_blocks.append(
                    UNetDownBlockCrossAttn(input_channel, output_channel, self.time_hidden_dim, act=self.conv_act, add_downsample=add_downsample, context_dim=self.context_dim)
                )
            else:
                self.encoder_blocks.append(
                    UNetDownBlock(input_channel, output_channel, self.time_hidden_dim, act=self.conv_act, add_downsample=add_downsample)
                )
            if not i == self.n_layers -1:
                ds *= 2
            
        self.mid_block = UNetMidBlock(
            in_channels=block_out_channels[-1],
            output_channels=block_out_channels[-1],
            act=self.conv_act,
            context_dim=self.context_dim,
            time_embed_dim=self.time_hidden_dim,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(self.n_layers):
            # if i == 0:
            #     prev_output_channel = output_channel
            #     input_channel = output_channel
            #     output_channel = reversed_block_out_channels[i]
            # else:
            #     prev_output_channel = output_channel
            #     input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            #     output_channel = reversed_block_out_channels[i]
            skip_channels = reversed_block_out_channels[i]
            output_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            down_channels = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            
            add_upsample=True
            if i == 0:
                add_upsample = False

            if ds in attention_resolutions:
                self.decoder_blocks.append(
                    UNetUpBlockCrossAttn(skip_channels, down_channels, output_channel, self.time_hidden_dim, act=self.conv_act,add_upsample=add_upsample, context_dim=self.context_dim)
                )
            else:
                self.decoder_blocks.append(
                    UNetUpBlock(skip_channels, down_channels, output_channel, self.time_hidden_dim, act=self.conv_act,add_upsample=add_upsample)
                )
            ds /= 2
        num_groups_out = 32
        norm_eps = 1e-5
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps, context=None, use_cfg=False):
        device = x.device
        b, c, l, w = x.shape
        dtype=torch.float32
        
        if self.conditioning == 'text':
            if not self.text_tokenizer._device == device:
                self.text_tokenizer.to(device)
            context =  self.text_tokenizer.get_embeddding(context)
            context = context.unsqueeze(1)
            if use_cfg:
                drop_context = torch.rand(b, device=device) < self.cfg_prob
                context[drop_context, :] = self.empty_context
        else:
            context = None
        
        timesteps = timesteps.view(-1,1)
        t = self.time_mlp(timesteps)

        x = self.conv_in(x)
        skip_connections = [x]
        for i in range(self.n_layers):
            x = self.encoder_blocks[i](x, t, context=context)
            skip_connections.append(x)
        
        x = self.mid_block(x, t, context=context)
        skip_connections = skip_connections[:-1] # remove the last skip_connection, as we will use the mid_block processed features instead

        for i in range(self.n_layers):
            skip_connection = skip_connections[-1]
            skip_connections = skip_connections[:-1]
            x = self.decoder_blocks[i](x, skip_connection, t, context=context)

        x = self.conv_norm_out(x)
        x = self.conv_out(x)
        return x
    
if __name__ == '__main__':
    device='cuda:2'
    model = UNet(3, 3, [64,128,256,512], 128).to(device)
    inp = torch.randn((3,3,256,256)).to(device)
    timesteps = torch.randint(0, 100, (3,1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.MSELoss()
    for i in range(2000):
        o = model(inp, timesteps)
        loss = loss_fn(o, inp)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item(), torch.max(o).item(), torch.min(o).item())
