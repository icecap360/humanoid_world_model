#!/bin/bash  
export CUDA_VISIBLE_DEVICES=1,2
# python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py
python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py --config-name=flow_video_mmdit
