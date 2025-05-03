#!/bin/bash
set -e
SNR=-3.0          # change here when you test another SNR

echo "▶  VCTK"
python3 verify_dataset.py \
        --clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
        --noisy_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
        --samples 5 \
        --snr "$SNR"

echo "▶  TIMIT"
python3 verify_dataset.py \
        --clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
        --noisy_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
        --samples 5 \
        --snr "$SNR"