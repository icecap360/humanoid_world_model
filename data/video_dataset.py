import json
import math
import os
import cv2
from pathlib import Path

import numpy as np
import torch
from einops import pack
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import default_collate
import multiprocessing as mp

_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

class RawVideoDataset(TorchDataset):
    def __init__(self, data_dir, cfg, vae, n_input=8, n_output=1, stride=4, skip_frame=1, with_actions = True):
        super().__init__()
        # mp.set_start_method('spawn')

        self.data_dir = Path(data_dir)
        self.vae = vae
        self.cfg= cfg
        self.image_size = cfg.image_size
        
        # Load main metadata
        with open(self.data_dir / "metadata.json") as f:
            metadata = json.load(f)
            self.num_shards = metadata["num_shards"]
            self.query = metadata["query"]
            self.hz = metadata["hz"]
            self.num_images = metadata["num_images"]
        
        # Load shard-specific metadata
        self.shard_sizes = []
        for shard in range(self.num_shards):
            with open(self.data_dir / f"metadata/metadata_{shard}.json") as f:
                shard_metadata = json.load(f)
                self.shard_sizes.append(shard_metadata["shard_num_frames"])
        
        # Calculate cumulative shard sizes for index mapping
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
        assert self.cumulative_sizes[-1] == self.num_images, "Metadata mismatch in total number of frames"
        
        # Store video paths instead of keeping captures open
        self.video_paths = [
            str(self.data_dir / f"videos/video_{shard}.mp4")
            for shard in range(self.num_shards)
        ]
        
        # Store action paths, and open them up if with_actions
        self.action_paths = [
            str(self.data_dir / f"states/states_{shard}.bin")
            for shard in range(self.num_shards)
        ]
        self.with_actions = with_actions
        if self.with_actions:
            self.action_shards = []
            for i,path in enumerate(self.action_paths):
                action_shard = np.memmap(path, dtype=np.float32, mode="r", shape=(self.shard_sizes[i], 25))
                self.action_shards.append(
                    action_shard
                )
            self.action_shards = np.concatenate(self.action_shards, 0)
            self.action_shards = (self.action_shards - np.mean(self.action_shards, 0)) / np.std(self.action_shards, 0)
        
        # Store action paths, and open them up if with_actions
        self.segment_paths = [
            str(self.data_dir / f"segment_idx/segment_idx_{shard}.bin")
            for shard in range(self.num_shards)
        ]
        
        segment_shards = []
        for i,path in enumerate(self.segment_paths):
            segment_shard = np.memmap(path, dtype=np.int32, mode="r", shape=(self.shard_sizes[i], 1))
            segment_shards.append(segment_shard)
        self.segment_shards = np.concatenate(segment_shards).squeeze(-1)
        
        # Compute the valid start indexes            
        self.n_input, self.n_output, self.stride, self.skip_frame = n_input, n_output, stride, skip_frame
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.n_output + self.n_input) # * self.skip_frame 
        
        start_indices = np.arange(0, self.num_images - self.video_len, self.stride)
        end_indices = start_indices + self.video_len
        
        start_segment_ids = self.segment_shards[start_indices] # 
        end_segment_ids = self.segment_shards[end_indices] # 
        
        self.valid_start_inds = start_indices[start_segment_ids == end_segment_ids]

        # Verify all video files exist and are readable
        for path in self.video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise IOError(f"Could not open video file: {path}")
            cap.release()
        
        # Verify all action files exist and are readable
        for path in self.action_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")

    def get_index_info(self, idx):
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        frame_idx = idx - self.cumulative_sizes[shard_idx]
        return shard_idx, frame_idx 
    
    def extract_frames_opencv(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Jump to start frame
        
        frames = [] # [np.zeros((self.image_size, self.image_size, 3)) for _ in range(end_frame-start_frame)]
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame)
            # frames[frame_idx - start_frame] = frame
        
        cap.release()
        return np.array(frames)  # Shape: (num_frames, H, W, 3)

    def __len__(self):
        return len(self.valid_start_inds)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.valid_start_inds):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.num_images} images")

        start_idx = self.valid_start_inds[idx]
        end_idx = start_idx + self.video_len
        start_shard_idx, start_frame_idx = self.get_index_info(start_idx)
        end_shard_idx, end_frame_idx = self.get_index_info(end_idx)
        
        assert self.segment_shards[start_idx] == self.segment_shards[end_idx] 
        shard_idx = start_shard_idx
        
        frames = self.extract_frames_opencv(
            self.video_paths[shard_idx],
            start_frame_idx, 
            end_frame_idx
        )
        
        frames = np.moveaxis(frames, 3, 0)
        frames = frames/_UINT8_MAX_F * 2.0 - 1.0
        
        past_frames = frames[:, :self.n_input]
        future_frames = frames[:, self.n_input:]
        assert future_frames.shape[1] == self.n_output
        
        ret = {
            "past_frames": past_frames.astype(np.float32), 
            "future_frames": future_frames.astype(np.float32)
            }

        if self.with_actions:
            ret["past_actions"] = self.action_shards[start_idx:start_idx+self.n_input].astype(np.float32)
            ret["future_actions"] = self.action_shards[start_idx + self.n_input:start_idx + self.n_input + self.n_output].astype(np.float32)
        
        return ret

    def get_frame_info(self, idx):
        """Helper method to debug frame locations"""
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        frame_idx = idx - self.cumulative_sizes[shard_idx]
        return {
            "global_index": idx,
            "shard_index": shard_idx,
            "frame_index": frame_idx,
            "video_path": self.video_paths[shard_idx]
        }

def RawVideoDataset_collate_fn(cfg, vae, samples):
    '''
    We encode the batches in the collate_fn to speed up the dataloader
    '''
    # Extract past and future frames from each sample
    past_frames_list = np.stack([s['past_frames'] for s in samples], 0)
    future_frames_list = np.stack([s['future_frames'] for s in samples], 0)
    past_actions_list = np.stack([s['past_actions'] for s in samples], 0)
    future_actions_list = np.stack([s['future_actions'] for s in samples], 0)
    
    # Create batch dictionary
    batch = {
        'past_frames': torch.from_numpy(past_frames_list).to(vae._dtype),
        'future_frames': torch.from_numpy(future_frames_list).to(vae._dtype),
        'past_actions': torch.from_numpy(past_actions_list).to(vae._dtype),
        'future_actions': torch.from_numpy(future_actions_list).to(vae._dtype)
    }
    
    # Encode the batch using the provided encode_batch function
    batch = encode_video_batch(cfg, batch, vae)
    return batch

def encode_video_batch(cfg, batch, vae):
    orig_dtype = batch['past_frames'].dtype
    past_frames = batch['past_frames'].to(vae._dtype)
    future_frames = batch['future_frames'].to(vae._dtype)
    device = next(vae.parameters()).device
    past_latents, _ = create_condition_latent(
        vae, 
        past_frames, 
        cfg.conditioning.num_past_frames,
        cfg.conditioning.num_future_frames,
        cfg.conditioning.num_past_latents,
        cfg.conditioning.num_future_latents,
        device
    )
    _, future_latents = create_label_latent(
        vae, 
        past_frames, 
        future_frames,
        cfg.conditioning.num_past_latents,
        cfg.conditioning.num_future_latents,
        device)
    batch['past_latents'] = past_latents.to(orig_dtype)
    batch['future_latents'] = future_latents.to(orig_dtype)
    return batch

def create_condition_latent(tokenizer, past_frames, num_past_frames, num_future_frames, num_past_latents, num_future_latent, device):
    B, C, T, H, W = past_frames.shape
    
    padding_frames = past_frames.new_zeros(B, C, num_future_frames, H, W)
    encode_past_frames = torch.cat([past_frames, padding_frames], dim=2)
    (latent, ) = tokenizer.encode(encode_past_frames.to(device))

    past_latent = latent[:, :, :num_past_latents]
    future_latent = latent[:, :, num_past_latents:]
    assert future_latent.shape[2] == num_future_latent
    return past_latent, future_latent

def create_label_latent(tokenizer, past_frames, future_frames, num_past_latents, num_future_latent, device):
    B, C, T, H, W = past_frames.shape
    all_frames = torch.concatenate((past_frames, future_frames), 2) 
    (latent, ) = tokenizer.encode(all_frames.to(device))
    past_latent = latent[:, :, :num_past_latents]
    future_latent = latent[:, :, num_past_latents:]
    assert future_latent.shape[2] == num_future_latent
    return past_latent, future_latent

if  __name__ == '__main__':
    dataset = RawVideoDataset(
        data_dir="/pub0/qasim/1xgpt/data/data_v2_raw/train_v2.0_raw",
        n_input=8,
        n_output=1,
        stride=4,
        skip_frame=1,
    )
    from cosmos_tokenizer.networks import TokenizerConfigs
    from cosmos_tokenizer.image_lib import ImageTokenizer
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    tokenizer_path = "/pub0/qasim/1xgpt/Cosmos-Tokenizer/pretrained_ckpts"
    spatial_compression = 8 #cfg.image_tokenizer.spatial_compression
    temporal_compression = 4 # cfg.image_tokenizer.temporal_compression
    tokenizer_type = 'CV' # cfg.image_tokenizer.tokenizer_type

    tokenizer_config = TokenizerConfigs["CV"].value
    tokenizer_config.update(dict(spatial_compression=spatial_compression))
    if "I" in tokenizer_type:
        model_name = f"Cosmos-Tokenizer-{tokenizer_type}{spatial_compression}x{spatial_compression}"
    else:
        model_name = f"Cosmos-Tokenizer-{tokenizer_type}{temporal_compression}x{spatial_compression}x{spatial_compression}"
    
    vae = CausalVideoTokenizer(
            checkpoint=Path(tokenizer_path) / model_name / "autoencoder.jit",
            checkpoint_enc=Path(tokenizer_path) / model_name / "encoder.jit",
            checkpoint_dec=Path(tokenizer_path) / model_name / "decoder.jit",
            tokenizer_config=tokenizer_config,
            device=None,
            dtype="bfloat16",
        ).to('cuda')
    input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)
    (latent, ) = vae.encode(input_tensor)
    print(latent.shape)

    print(len(dataset))
    print(dataset[0])
    print(dataset[6050])
    print('Compeleted')
    vae.autoencode()