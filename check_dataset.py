#!/usr/bin/env python3
"""
Check if dataset directories contain the expected WAV files.
"""

import os
import sys
from pathlib import Path

def count_wav_files(directory):
    """Count the number of WAV files in a directory (recursive)."""
    if not os.path.exists(directory):
        return 0, f"Directory doesn't exist: {directory}"
    
    try:
        wav_files = list(Path(directory).rglob('*.wav'))
        return len(wav_files), None
    except Exception as e:
        return 0, f"Error scanning directory {directory}: {str(e)}"

def check_dataset_structure():
    """Check if dataset directories contain WAV files."""
    # Dataset directories from config
    dataset_dirs = {
        "Clean Training": "./Dataset/clean_trainset",
        "Clean Validation": "./Dataset/clean_valset",
        "Clean Test": "./Dataset/clean_testset", 
        "Noisy Training": "./Dataset/noisy_trainset",
        "Noisy Validation": "./Dataset/noisy_valset",
        "Noisy Test": "./Dataset/noisy_testset"
    }
    
    # Also check SNR-specific directories for test set
    snr_levels = [-6, -3, 0, 3, 6, 9, 12, 15]
    for snr in snr_levels:
        snr_dir = f"./Dataset/noisy_testset/snr_{snr}dB"
        dataset_dirs[f"Noisy Test (SNR {snr}dB)"] = snr_dir
    
    # Check each directory
    print("Checking dataset structure...")
    print("-" * 70)
    
    all_empty = True
    total_files = 0
    
    for name, directory in dataset_dirs.items():
        count, error = count_wav_files(directory)
        status = f"{count} WAV files" if error is None else f"ERROR: {error}"
        print(f"{name:25}: {status}")
        
        if count > 0:
            all_empty = False
            total_files += count
    
    print("-" * 70)
    print(f"Total WAV files found: {total_files}")
    
    # Provide recommendations based on findings
    if all_empty:
        print("\nALL DIRECTORIES ARE EMPTY OR DO NOT EXIST")
        print("\nRecommendations:")
        print("1. Run the dataset preparation script: ./run_dataset.sh")
        print("2. Check dataset paths in config/config.yaml and config/speech_dataset.yaml")
        print("3. Verify that you have sufficient disk space for dataset creation")
    elif total_files == 0:
        print("\nNO WAV FILES FOUND IN ANY DIRECTORY")
        print("\nRecommendations:")
        print("1. Check file formats - ensure files are saved as .wav")
        print("2. Verify file permissions")
    else:
        # If there are some files but some directories are empty
        empty_dirs = [name for name, dir in dataset_dirs.items() 
                    if os.path.exists(dir) and count_wav_files(dir)[0] == 0]
        
        if empty_dirs:
            print("\nSome directories exist but contain no WAV files:")
            for dir_name in empty_dirs:
                print(f"- {dir_name}")
            
            print("\nRecommendations:")
            print("1. Run the dataset preparation script again to ensure all directories are populated")
            print("2. Check if dataset creation completed successfully")
    
    # Return True if there are WAV files in all required directories
    main_dirs = ["Clean Training", "Clean Validation", "Noisy Training", "Noisy Validation"]
    return all(count_wav_files(dataset_dirs[d])[0] > 0 for d in main_dirs)

if __name__ == "__main__":
    if check_dataset_structure():
        print("\nDataset structure looks good! All main directories contain WAV files.")
        sys.exit(0)
    else:
        print("\nDataset structure has issues. Please fix before attempting to train.")
        sys.exit(1)