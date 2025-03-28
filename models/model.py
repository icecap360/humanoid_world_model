import abc
import torch.nn as nn
import torch 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    @abc.abstractmethod
    def forward(self, batch, timesteps, device, use_cfg=False):
        pass
    
class ImageModel(Model):
    def __init__(self):
        super().__init__()
    def extract_fields_imgs(self, batch, use_cfg, device):
        '''
        Extracts the noisy_latents and captions from the batch
        '''
        if self.conditioning == 'text':
            latents = batch['noisy_latents']
            context = batch.get('captions')
            batch_size = latents.shape[0]
            # if not self.text_tokenizer._device == device:
            #     self.text_tokenizer.to(device)
            if context is None:
                context = self.empty_context.repeat(batch_size,1,1)
            else:
                context =  self.conditioning_manager['text'].get_embeddding(context)
                context = context.unsqueeze(1)
                if use_cfg:
                    drop_context = torch.rand(batch_size,    device=device) < self.cfg_prob
                    context[drop_context, :] = self.empty_context
            return latents, context
        else:
            latents = batch['latents']
            context = None
            return latents, context