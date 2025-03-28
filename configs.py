from dataclasses import dataclass
import torch 

@dataclass
class BaseConfig:
    image_size = 192  # the generated image resolution
    train_batch_size = 264 # 32
    eval_batch_size = 64  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 3
    learning_rate = 8e-5 # 5e-5
    lr_warmup_steps = 100 # 500
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    eval_iters = 49
    save_model_iters = 5000

    conditioning = 'text'
    prompt_file= "sample_prompts.txt"
    text_tokenizer="CLIP"    
    
    noise_steps = 1000
    cfg_prob = 0.2
    eval_dir = "/pub0/qasim/1xgpt/humanoid_world_model/logs"  # the model name locally and on the HF Hub
    results_dir = "/pub0/qasim/1xgpt/humanoid_world_model/results"  # the model name locally and on the HF Hub
    log_dir = "/pub0/qasim/1xgpt/humanoid_world_model/logs"  # the model name locally and on the HF Hub


    # push_to_hub = True  # whether to upload the saved model to the HF Hub
    # hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    # hub_private_repo = False
    # overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    tokenizer_type = "CI" # ["CI", "DI"]
    spatial_compression = 8 # []8, 16
    dtype = "bfloat16"
    eval_device = 'cuda:0'
    pretrained_models = "/pub0/qasim/1xgpt/Cosmos-Tokenizer/pretrained_ckpts"
    resume_train = False
    resume_model = "/pub0/qasim/1xgpt/humanoid_world_model/val/12-26-12-49/checkpoint-0-5"  
    exp_prefix = 'DDPM'
    unet_blocks = [256, 256*2, 256*3, 256*4] # [128,256,512, 1024]
    unet_attention_resolutions = [32, 16, 8, 4]
    
    data = {
        "type" : "coco",
        "hmwm_train_dir": "/pub0/qasim/1xgpt/data/data_v2_raw/train_v2.0_raw",
        "hmwm_val_dir": "/pub0/qasim/1xgpt/data/data_v2_raw/val_v2.0_raw",
        "coco_train_imgs": "/pub0/data/mscoco_2017/coco/images/train2017",
        "coco_train_ann": "/pub0/data/mscoco_2017/coco/annotations/captions_train2017.json",
        "coco_val_imgs": "/pub0/data/mscoco_2017/coco/images/val2017",
        "coco_val_ann": "/pub0/data/mscoco_2017/coco/annotations/captions_val2017.json"
    }

_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)


if __name__ == "__main__":
    config = BaseConfig()
