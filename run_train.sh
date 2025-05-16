#!/bin/bash

# Set paths and environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=0

# Default configuration file path
CONFIG_PATH="./config/config.yaml"

# Create checkpoint directory if it doesn't exist
mkdir -p ./Checkpoints

# Run training
echo "Starting BCCTN training..."
python train.py

echo "Training complete!"