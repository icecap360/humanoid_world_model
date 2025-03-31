from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from einops import pack, unpack

from .rawimage_dataset import RawImageDataset
from .video_dataset import RawVideoDataset, RawVideoDataset_collate_fn, encode_video_batch
from .coco_dataset import CustomCoco
from functools import partial

def get_dataloaders(
    data_type,
    cfg,
    hmwm_train_dir,
    hmwm_val_dir,
    coco_train_imgs,
    coco_train_ann,
    coco_val_imgs,
    coco_val_ann,
    conditioning_type,
    conditioning_manager,
    image_size,
    train_batch_size,
    val_batch_size,
    num_past_frames=None,
    num_future_frames=None,
    vae=None,
):
    """Factory function to return train and validation dataloaders based on the dataset type."""
    if data_type == "1xgpt_image":
        assert conditioning_type != "text", "Conditioning must not be 'text' for 1xgpt dataset."
        train_dataset = RawImageDataset(hmwm_train_dir)
        val_dataset = RawImageDataset(hmwm_val_dir)
    elif data_type == "1xgpt_video":
        assert conditioning_type != "text", "Conditioning must not be 'text' for 1xgpt dataset."
        assert num_past_frames != None
        assert num_future_frames != None
        with_action = False
        if conditioning_type == 'action':
            with_action = True
        train_dataset = RawVideoDataset(hmwm_train_dir,
                                        cfg,
                                        vae,
                                        n_input=num_past_frames,
                                        n_output=num_future_frames,
                                        with_actions=with_action,
                                        stride=num_past_frames // 2)
        val_dataset = RawVideoDataset(hmwm_val_dir,
                                    cfg,
                                    vae,
                                    n_input=num_past_frames,
                                    n_output=num_future_frames,
                                    with_actions=with_action,
                                    stride=num_past_frames // 2)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
            # collate_fn=partial(RawVideoDataset_collate_fn, cfg, vae),
            **get_dataloader_kwargs())
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, 
            # collate_fn=partial(RawVideoDataset_collate_fn, cfg, vae), 
            shuffle=False, **get_dataloader_kwargs())
        
        return train_dataloader, val_dataloader
    elif data_type.lower() == "coco":
        assert conditioning_type == "text", "Conditioning must be 'text' for COCO dataset."
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_dataset = CustomCoco(
            root=coco_train_imgs,
            annFile=coco_train_ann,
            text_tokenizer=conditioning_manager.get_module()['text'],
            transform=transform,
        )
        val_dataset = CustomCoco(
            root=coco_val_imgs,
            annFile=coco_val_ann,
            text_tokenizer=conditioning_manager.get_module()['text'],
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset type: {data_type}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
        # collate_fn=no_collate_fn,
        **get_dataloader_kwargs())
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, 
        # collate_fn=no_collate_fn, 
        shuffle=False, **get_dataloader_kwargs())
    
    return train_dataloader, val_dataloader

def get_dataloader_kwargs():
    """Get kwargs for DataLoader in distributed setting"""
    return {
        'pin_memory': True,
        'num_workers': 4,  # Increased from 0
        'prefetch_factor': 2,  # Added prefetch
        'persistent_workers': True,  # Keep workers alive between epochs
    }


def no_collate_fn(batch):
    return batch

def encode_batch(cfg, batch, vae, accelerator):
    if 'video' in cfg.gen_type.lower():
        batch = encode_video_batch(cfg, batch, vae)
        return batch['future_latents'], encode_video_batch(cfg, batch, vae)
    else:
        imgs = batch['imgs']
        with accelerator.autocast():
            imgs = imgs.to(getattr(torch, cfg.image_tokenizer.dtype))
            latents = vae.encode(imgs)[0]
        return latents, batch
