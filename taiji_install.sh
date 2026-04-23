#!/bin/bash
set -e

echo "==> [1/4] Installing LlamaFactory ..."
pip install -e .

echo "==> [2/4] Installing DeepSpeed ..."
# pip install -r requirements/deepspeed.txt
pip install deepspeed==0.15.4

echo "==> [3/4] Installing Flash Attention 2 ..."
pip install flash-attn --no-build-isolation
# echo "==> [3/4] Installing coscmd ..."
# pip install coscmd

echo "==> [4/4] Installing tensorboard ..."
pip install tensorboard

echo "==> All done!"