#!/bin/bash

# Set paths
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/DCNN/Checkpoints/Trained_model.ckpt"
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt" # current best
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/last.ckpt"
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/outputs/2025-05-11/16-34-17/logs/lightning_logs/version_0/checkpoints/epoch=94-step=91960.ckpt"
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/outputs/2025-05-12/15-30-55/logs/lightning_logs/version_0/checkpoints/epoch=89-step=87120.ckpt" 
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/outputs/2025-05-12/15-30-55/logs/lightning_logs/version_0/checkpoints/last.ckpt"
MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/outputs/2025-05-16/04-19-55/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt"
VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
VCTK_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset"
TIMIT_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset_timit"
TIMIT_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset_timit"
OUTPUT_DIR="./results_May04"  # Define output directory

# Create results directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Run paper-style evaluation with fixed SNR levels on VCTK dataset (matched condition)
echo "Running paper-style evaluation on VCTK dataset (matched condition)..."
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --vctk_test_dir "$VCTK_NOISY_DIR" \
    --vctk_clean_dir "$VCTK_CLEAN_DIR" \
    --timit_test_dir "$TIMIT_NOISY_DIR" \
    --timit_clean_dir "$TIMIT_CLEAN_DIR" \
    --output_dir "${OUTPUT_DIR}/vctk_paper_style" \
    --paper_style_eval \
    --batch_size 8 \
    --limit_pairs 750

# Step 3: Generate comparison table between our results and paper results
echo "Generating comparison table..."
CUDA_VISIBLE_DEVICES=5 python compare_results.py \
    --vctk_results "${OUTPUT_DIR}/vctk_paper_style/vctk_paper_style_results.csv" \
    --timit_results "${OUTPUT_DIR}/timit_paper_style/timit_paper_style_results.csv" \
    --output_file "${OUTPUT_DIR}/comparison_with_paper.csv"

echo "Evaluation complete! Results are saved in ${OUTPUT_DIR}"