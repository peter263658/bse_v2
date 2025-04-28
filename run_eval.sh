#!/bin/bash

# Set paths
MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/DCNN/Checkpoints/Trained_model.ckpt"    # Path to trained model
VCTK_TEST="/raid/R12K41024/BCCTN/Dataset/noisy_testset"                         # Path to VCTK noisy test set
VCTK_CLEAN="/raid/R12K41024/BCCTN/Dataset/clean_testset"                        # Path to VCTK clean test set
TIMIT_TEST="/raid/R12K41024/BCCTN/Dataset/noisy_testset_timit"                  # Path to TIMIT noisy test set
TIMIT_CLEAN="/raid/R12K41024/BCCTN/Dataset/clean_testset_timit"                 # Path to TIMIT clean test set
RESULTS_DIR="./results"                                                         # Directory to save results

# Create results directory
mkdir -p $RESULTS_DIR

# Run paper-style evaluation (with SNR-specific results)
python eval.py \
    --model_checkpoint $MODEL_CHECKPOINT \
    --vctk_test_dir $VCTK_TEST \
    --vctk_clean_dir $VCTK_CLEAN \
    --timit_test_dir $TIMIT_TEST \
    --timit_clean_dir $TIMIT_CLEAN \
    --output_dir $RESULTS_DIR/paper_style_eval \
    --paper_style_eval

# Run basic evaluation
python eval.py \
    --model_checkpoint $MODEL_CHECKPOINT \
    --test_dataset_path $VCTK_TEST \
    --clean_dataset_path $VCTK_CLEAN \
    --output_dir $RESULTS_DIR/basic_eval