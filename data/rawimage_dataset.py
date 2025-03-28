from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import json 
import numpy as np
import cv2
import bisect
import torch
from einops import rearrange
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)


class RawImageDataset(TorchDataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = Path(data_dir)
        
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
        
        # Verify all video files exist and are readable
        for path in self.video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise IOError(f"Could not open video file: {path}")
            cap.release()

    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.num_images} images")
        
        # Find which shard contains this index
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        
        # Calculate frame index within the shard
        frame_idx = idx - self.cumulative_sizes[shard_idx]
        
        # Open video file for this specific read
        cap = cv2.VideoCapture(self.video_paths[shard_idx])
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_paths[shard_idx]}")
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise IOError(f"Failed to read frame {frame_idx} from shard {shard_idx}")
        
        frame = np.moveaxis(frame, 2, 0)
        frame = frame/_UINT8_MAX_F * 2.0 - 1.0

        return {
            "imgs": frame
        }
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

# class RawImageDataset(TorchDataset):
#     def __init__(
#         self,
#         data_dir
#     ):
#         super().__init__()
#         data_dir = Path(data_dir)
#         with open(data_dir / "metadata.json") as f:
#             metadata = json.load(f)
#             self.num_shards = metadata["num_shards"]
#             self.query = metadata["query"]
#             self.hz = metadata["hz"]
#             self.num_images = metadata["num_images"]
        
#         shard_num_images = []
#         for shard in range(self.num_shards):
#             with open(data_dir / f"metadata/metadata_{shard}.json") as f:
#                 shard_metadata = json.load(f)                
#                 shard_num_images.append(shard_metadata["shard_num_frames"])
#         shard_num_images = np.cumsum(shard_num_images)
#         self.shard_num_images = np.cumsum(shard_num_images)
        
#         self.data_dir = data_dir
#         assert shard_num_images[-1] == self.num_images

#         self.caps, self.video_paths = [], []
#         for shard in range(self.num_shards):
#             video_path = data_dir / f"videos/video_{shard}.mp4"
#             self.caps.append(cv2.VideoCapture(video_path))
#             self.video_paths.append(video_path)
        
#     def __len__(self):
#         return self.num_images
    
#     def __getitem__(self, index):
#         indices =  np.where(self.shard_num_images > index)[0]
#         if len(indices) == 0:
#             shard_index = 0
#             frame_index = index
#         else:
#             shard_index = indices[0]
#             divisor = self.shard_num_images[indices[0] - 1]
#             frame_index = index % divisor
        
#         self.caps[shard_index].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#         ret, frame = self.caps[shard_index].read()

#         # cap = cv2.VideoCapture(self.video_paths[shard_index])
#         # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#         # ret, frame = cap.read()
#         # cap.release()
#         return frame
    
#     def __del__(self):
#         for i in range(len(self.caps)):
#             self.caps[i].release()
    
#     def close(self):
#         for i in range(len(self.caps)):
#             self.caps[i].release()

if __name__ == '__main__':
    dataset = RawImageDataset('/pub0/qasim/1xgpt/data/data_v2_raw/train_v2.0_raw')
    import time, random
    print('RawImageDataset')
    for _ in range(10):
        indexes = random.sample(range(0, len(dataset)), 64)
        assert len(indexes) == 64
        start = time.time()
        for i in indexes:
            x = dataset[i]
        end = time.time()
        print(end - start)