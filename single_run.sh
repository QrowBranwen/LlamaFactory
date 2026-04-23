#!/bin/zsh

TRAIN_CONFIG_FILE="${1:-examples/train_lora/ozpin/qwen35_27b_lora_sft_mix.yaml}"

FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29501 llamafactory-cli train "${TRAIN_CONFIG_FILE}"
