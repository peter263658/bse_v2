# Binaural Speech Enhancement Using Deep Complex Convolutional Transformer Networks (BCCTN)

This repository contains the implementation of the paper: "Binaural Speech Enhancement Using Deep Complex Convolutional Transformer Networks" by Vikas Tokala et al. (2024).

## Overview

The BCCTN model enhances binaural speech signals in noisy environments while preserving spatial cues. The model uses complex-valued convolutional neural networks with an encoder-decoder architecture and a complex multi-head attention transformer.

![BCCTN Architecture](path/to/architecture_diagram.png)

## Key Features

- Complex-valued convolutional and transformer neural network
- Binaural processing with individual complex ratio masks for left and right channels
- Novel loss function that preserves spatial information (ILD and IPD)
- Evaluation metrics for both speech intelligibility and spatial cue preservation

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Additional dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BCCTN.git
cd BCCTN

# Create a virtual environment
python -m venv bcctn_env
source bcctn_env/bin/activate  # On Windows: bcctn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

To prepare the training and evaluation datasets as described in the paper:

```bash
# For VCTK dataset (training and matched condition testing)
python data_preparation.py \
  --clean_dir /path/to/vctk \
  --noise_dir /path/to/noisex92 \
  --hrir_path /path/to/hrir \
  --output_dir ./Dataset \
  --hrir_format wav \
  --dataset_type vctk

# For TIMIT dataset (unmatched condition testing)
python data_preparation.py \
  --clean_dir /path/to/timit \
  --noise_dir /path/to/noisex92 \
  --hrir_path /path/to/hrir \
  --output_dir ./Dataset \
  --hrir_format wav \
  --dataset_type timit
```

This creates:
- Clean and noisy binaural training sets
- Clean and noisy binaural validation sets
- Clean and noisy binaural test sets for both VCTK and TIMIT
- For TIMIT, separate directories for each SNR level (-6, -3, 0, 3, 6, 9, 12, 15 dB)

## Configuration

Configuration files are in the `config/` directory:
- `config.yaml`: Main configuration
- `model.yaml`: Model architecture settings
- `training.yaml`: Training parameters
- `dataset/speech_dataset.yaml`: Dataset paths

Adjust these files to match your environment and needs.

## Training

To train the BCCTN model:

```bash
python train.py
```

The training script:
- Loads the dataset according to the configuration
- Creates the complex-valued binaural model
- Sets up the combined loss function (SNR, STOI, ILD, IPD)
- Trains the model with regular evaluation
- Saves model checkpoints in the `logs/` directory

## Evaluation

To evaluate a trained model:

```bash
# Basic evaluation
python evaluate.py \
  --checkpoint logs/lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt \
  --test_dir ./Dataset/noisy_testset \
  --clean_dir ./Dataset/clean_testset \
  --output_dir ./evaluation_results

# Evaluation with specific SNR
python evaluate.py \
  --checkpoint logs/lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt \
  --test_dir ./Dataset/noisy_testset \
  --clean_dir ./Dataset/clean_testset \
  --output_dir ./evaluation_results \
  --snr 0

# Paper-style evaluation (all SNRs, both VCTK and TIMIT)
python evaluate.py \
  --checkpoint logs/lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt \
  --test_dir ./Dataset/noisy_testset \
  --clean_dir ./Dataset/clean_testset \
  --timit_test_dir ./Dataset/noisy_testset_timit \
  --timit_clean_dir ./Dataset/clean_testset_timit \
  --output_dir ./evaluation_results \
  --paper_style
```

The evaluation generates:
- Metrics: MBSTOI, SegSNR improvement, ILD error, IPD error
- Enhanced audio files
- Spectrograms and visualizations
- Paper-style tables for comparison with the original paper results

## Model Architecture

The BCCTN model consists of:

1. **Encoder**: 6 complex convolutional layers with PReLU activation
2. **Transformer**: Complex-valued multi-head attention with 32 heads
3. **Decoder**: 6 complex transposed convolutional layers
4. **Output**: Individual complex ratio masks for left and right ear channels

The model uses a combination of skip connections between encoder and decoder.

## Loss Function

The combined loss function consists of:

- **SNR Loss**: For noise reduction
- **STOI Loss**: For speech intelligibility improvement
- **ILD Loss**: For preserving Interaural Level Differences
- **IPD Loss**: For preserving Interaural Phase Differences

## Citation

```
@inproceedings{tokala2024binaural,
  title={Binaural Speech Enhancement using Deep Complex Convolutional Transformer Networks},
  author={Tokala, Vikas and Grinstein, Eric and Brookes, Mike and Doclo, Simon and Jensen, Jesper and Naylor, Patrick A},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
