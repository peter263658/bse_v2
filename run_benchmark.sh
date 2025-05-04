#!/bin/bash

# Set paths to model checkpoint and test data
MODEL_CHECKPOINT="/raid/R12K41024/BCCTN/DCNN/Checkpoints/Trained_model.ckpt"  # Adjust path as needed
INPUT_FILE="/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/p274_p274_111_az55_snr+0.0.wav"  # Choose a single file from test set
OUTPUT_DIR="./benchmark_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run benchmark
echo "Running BCCTN benchmark on CPU with a single 2-second pattern"
echo "============================================================="

python benchmark_bcctn.py \
    --model_checkpoint $MODEL_CHECKPOINT \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --num_runs 100

echo ""
echo "Benchmark complete. Results saved to $OUTPUT_DIR"
echo "See $OUTPUT_DIR/benchmark_results.txt for detailed results"
