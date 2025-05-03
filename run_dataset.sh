#!/bin/bash

# Set paths
CLEAN_DIR_VCTK="/home/R12K41024/Dataset/VCTK-Corpus/VCTK-Corpus/wav48"  # Path to directory containing VCTK and TIMIT folders
CLEAN_DIR_TIMIT="/home/R12K41024/Dataset/TIMIT/timit/timit"             # Path to directory containing VCTK and TIMIT folders
NOISE_DIR="/home/R12K41024/Dataset/Noises-master/NoiseX-92"             # Path to NOISEX-92 noise database
HRIR_PATH="/home/R12K41024/Dataset/HRIR_database_wav/hrir/anechoic"     # Path to HRIR files
OUTPUT_DIR="/raid/R12K41024/BCCTN/Dataset"                              # Base path where to save the dataset

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Check if prepare_data.py exists
if [ ! -f "prepare_data.py" ]; then
    echo "Error: prepare_data.py not found in current directory"
    exit 1
fi

# Create VCTK dataset with SNR subdirectories
echo "Creating VCTK dataset..."
python prepare_data.py \
    --clean_dir "${CLEAN_DIR_VCTK}" \
    --noise_dir "${NOISE_DIR}" \
    --hrir_path "${HRIR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --hrir_format wav \
    --dataset_type vctk \
    --use_snr_subdirs true

# Create TIMIT dataset for unmatched condition testing with SNR subdirectories
echo "Creating TIMIT dataset..."
python prepare_data.py \
    --clean_dir "${CLEAN_DIR_TIMIT}" \
    --noise_dir "${NOISE_DIR}" \
    --hrir_path "${HRIR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --hrir_format wav \
    --dataset_type timit \
    --use_snr_subdirs true

echo "Dataset preparation complete!"