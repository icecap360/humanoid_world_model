# Generating Humanoid Futures: Multimodal Joint Attention Between Past and Future for Action-Guided Video Prediction

Author: **Qasim Ali**  
License: **MIT License**  

---

## Overview

This repository contains the implementation of a world model for humanoid robots, designed to generate future video frames conditioned on past video frames, past actions, and future actions. The project is titled **"Generating Humanoid Futures: Multimodal Joint Attention Between Past and Future for Action-Guided Video Prediction"** and will be submitted to the Humanoid Workshop at CVPR 2025.

The model leverages **flow matching** for training and incorporates **joint attention** mechanisms inspired by image generation techniques (e.g., Stable Diffusion 3). These mechanisms are adapted to video generation, replacing the traditional two-stage attention with a more parameter-efficient joint attention scheme. The goal is to reduce the parameter count while maintaining high performance.

The dataset used for training was provided by the **1xgpt contest**, and we extend our gratitude to 1xgpt for making this data available.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url/humanoid_world_model.git
   cd humanoid_world_model
   ```

2. Create a new Conda environment:
   ```bash
   conda create -n humanoid_world_model python=3.8 -y
   conda activate humanoid_world_model
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the cosmos tokenizer from the [repo](https://github.com/NVIDIA/Cosmos-Tokenizer).

---

## Usage

The repository includes several scripts for training, evaluation, and debugging. Below are the instructions for running these scripts. The configurations for debugging and training are defined in `.vscode/launch.json`.

### 1. **Training the Model**
   To train the model, use the `train.py` script:
   ```bash
   python train.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "train (one sample)",
       "type": "debugpy",
       "request": "launch",
       "program": "train.py",
       "console": "integratedTerminal",
       "args": ["one_sample=True"],
       "justMyCode": false,
       "env": {
           "CUDA_VISIBLE_DEVICES": "0"
       }
   }
   ```

### 2. **Evaluating the Model**
   To evaluate the model, use the `eval_diffusion.py` script:
   ```bash
   python eval_diffusion.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "eval",
       "type": "debugpy",
       "request": "launch",
       "program": "eval_diffusion.py",
       "console": "integratedTerminal"
   }
   ```

### 3. **Profiling the Model**
   To profile the model, use the `profile_model.py` script:
   ```bash
   python profile_model.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "profiler",
       "type": "debugpy",
       "request": "launch",
       "program": "profile_model.py",
       "console": "integratedTerminal"
   }
   ```

### 4. **Distributed Training**
   For distributed training, use the `torch.distributed.run` module:
   ```bash
   python -m torch.distributed.run --nproc_per_node=2 train.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "train (distributed + debug)",
       "type": "python",
       "request": "launch",
       "module": "torch.distributed.run",
       "console": "integratedTerminal",
       "args": [
           "--nproc_per_node=1",
           "--rdzv_backend", 
           "c10d",
           "--rdzv_endpoint", 
           "localhost:29500",
           "--nnodes=1",
           "train.py",
           "debug=True"
       ],
       "justMyCode": false,
       "env": {
           "CUDA_VISIBLE_DEVICES": "1,2",
           "TORCH_NCCL_DEBUG": "INFO",
           "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
           "NCCL_P2P_DISABLE": "0",
           "NCCL_ASYNC_ERROR_HANDLING": "1"
       }
   }
   ```

---

## Project Structure

- **`models/`**: Contains the implementation of the video generation models, including joint attention mechanisms and parameter reduction schemes.
- **`data/`**: Handles data loading and preprocessing for the 1xgpt dataset.
- **`configs/`**: YAML configuration files for training and evaluation.
- **`train.py`**: Main training script.
- **`eval_diffusion.py`**: Script for evaluating the model.
- **`profile_model.py`**: Script for profiling the model.
- **`.vscode/launch.json`**: Debug configurations for various scripts.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## Acknowledgments

- Special thanks to **1xgpt** for providing the dataset used in this project.
- This work is inspired by advancements in image generation (e.g., Stable Diffusion 3) and adapted for video generation tasks.

--- 

Thank you for exploring this repository! If you have any questions or suggestions, feel free to reach out.
