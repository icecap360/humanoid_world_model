# from datasets import load_dataset, DownloadConfig
# import aiohttp
# import fiftyone as fo

# download_config = DownloadConfig(
#     resume_download=True,          # Enable resuming of downloads if interrupted
#     max_retries=10,                # Increase number of retries
#     # download_timeout=100,         # Set the timeout to 2 hours (7200 seconds)
# )
# # dataset = load_dataset("HuggingFaceM4/COCO", cache_dir='/pub0/qasim/1xgpt/data/COCO', download_config=download_config )
# dataset = load_dataset("shunk031/MSCOCO",
#     year=2017,
#     coco_task="captions",
#       cache_dir='/pub0/data/mscoco_2017', download_config=download_config )
# fo.config.dataset_zoo_dir = '/pub0/qasim/1xgpt/data/COCO'

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

# Paths to images and annotation files
root = "/pub0/data/mscoco_2017/coco/images/train2017"  # Image directory
annFile = "/pub0/data/mscoco_2017/coco/annotations/captions_train2017.json"  # Annotation file

# Define transformation (resize, normalize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit model input (optional)
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create COCO Dataset
dataset = CocoDetection(root=root, annFile=annFile, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Test: Load one sample
image, target = dataset[0]
print("Image Shape:", image.shape)
print("Target:", target)  # Target contains annotations for the image
