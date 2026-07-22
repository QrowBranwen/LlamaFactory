#!/bin/bash
set -e

echo "==> [1/4] Installing LlamaFactory ..."
python -m pip install -e .

echo "==> [2/4] Installing DeepSpeed ..."
python -m pip install -r requirements/deepspeed.txt

# echo "==> [3/4] Installing Flash Attention 2 ..."
# pip install flash-attn --no-build-isolation
echo "==> [3/4] Installing coscmd ..."
python -m pip install coscmd

echo "==> [4/4] Installing WandB ..."
python -m pip install wandb

# CUDA 12.6
# python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
# python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> All done!"