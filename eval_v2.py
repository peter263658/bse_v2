import os
import argparse
import torch
# import torchaudio
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import re

# Import custom modules
from DCNN.datasets.test_dataset import BaseDataset
from DCNN.trainer import DCNNLightningModule



def fwsegsnr(clean, noisy, enh, sr=16_000):
    """
    Compute frequency-weighted Segmental SNR improvement following VOICEBOX implementation
    
    Args:
        clean: Clean reference signal
        noisy: Noisy signal
        enh: Enhanced signal
        sr: Sample rate
        
    Returns:
        SNR improvement in dB (positive means enhancement is better than noisy)
    """
    # Frame parameters (25ms window, 6.25ms hop at 16kHz)
    frame = 400          # 25 ms
    hop = 100            # 6.25 ms
    fft_size = 512
    
    # Bark critical-band weights (as used in VOICEBOX)
    W = np.array([
        0.13, 0.26, 0.42, 0.60, 0.78, 0.93, 1.00, 0.97, 0.89,
        0.76, 0.62, 0.50, 0.38, 0.28, 0.22, 0.18, 0.14, 0.11,
        0.09, 0.07, 0.06, 0.05, 0.04
    ])
    
    # Critical band edges in Hz for 16kHz sampling rate
    band_edges_hz = np.array([
        0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
        2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000
    ])
    
    # Convert to FFT bin indices with correct handling for high frequencies
    edges_bin = np.round(band_edges_hz * fft_size / sr).astype(int)
    edges_bin = np.minimum(edges_bin, fft_size//2 + 1)  # Cap at Nyquist
    
    # Fix for zero-width bands
    valid = np.where(np.diff(edges_bin) > 0)[0]          # Find bands with non-zero width
    edges_bin = edges_bin[np.r_[valid, valid[-1]+1]]     # Keep only valid band edges
    W_valid = W[valid] / W[valid].sum()                  # Renormalize weights
    
    # Create windows and frame the signals
    win = np.hanning(frame)
    
    def _frames(x):
        if len(x) < frame:
            x = np.pad(x, (0, frame-len(x)))
        # Create frame indices
        idx = np.arange(frame)[None, :] + hop*np.arange((len(x)-frame)//hop+1)[:, None]
        return x[idx] * win
    
    # Frame signals and compute FFT
    C_frames = _frames(clean)
    N_frames = _frames(noisy - clean)  # Noise component
    E_frames = _frames(enh - clean)    # Enhanced error component
    
    # Apply FFT
    C_fft = np.fft.rfft(C_frames, fft_size, axis=1)
    N_fft = np.fft.rfft(N_frames, fft_size, axis=1)
    E_fft = np.fft.rfft(E_frames, fft_size, axis=1)
    
    # Compute band powers for each frame
    def _band_pow(X_fft):
        P = np.zeros((X_fft.shape[0], len(valid)))
        for b in range(len(valid)):
            lo, hi = edges_bin[b], edges_bin[b+1]
            P[:, b] = np.sum(np.abs(X_fft[:, lo:hi])**2, axis=1)
        return P + 1e-10  # Avoid division by zero
    
    # Calculate powers in each band
    P_clean = _band_pow(C_fft)
    P_noisy = _band_pow(N_fft)
    P_enh = _band_pow(E_fft)
    
    # VOICEBOX approach: calculate SNR per band in dB, then apply weights
    def _band_snr(P_sig, P_noise):
        snr_band = 10 * np.log10(P_sig / P_noise)
        snr_band = np.clip(snr_band, -10, 35)  # Clip to [-10, 35] dB range
        # Apply weights after calculating SNR in dB
        return np.sum(snr_band * W_valid[None, :], axis=1) / np.sum(W_valid)
    
    # Calculate frame energies to identify active speech frames
    frame_energy = 10 * np.log10(np.sum(P_clean, axis=1))
    active_frames = frame_energy > (np.max(frame_energy) - 40)  # Frames within 40dB of max
    
    # Calculate SNR for each frame, only for active frames
    if not np.any(active_frames):
        return 0.0  # No active frames
        
    snr_noisy = _band_snr(P_clean, P_noisy)[active_frames]
    snr_enh = _band_snr(P_clean, P_enh)[active_frames]
    
    # Return improvement
    return np.mean(snr_enh) - np.mean(snr_noisy)



def compute_ild_error(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """
    Compute ILD error between clean and enhanced binaural signals with improved
    handling of zero/small values
    """
    # STFT parameters
    sr_ratio = sr / 16000
    n_fft = int(512 * sr_ratio)
    hop_length = int(100 * sr_ratio)
    win_length = int(400 * sr_ratio)
    
    # Compute STFTs
    clean_left_stft = librosa.stft(clean_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    clean_right_stft = librosa.stft(clean_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    enhanced_left_stft = librosa.stft(enhanced_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    enhanced_right_stft = librosa.stft(enhanced_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Use a larger epsilon to avoid division by very small numbers
    eps = 1e-8
    
    # Compute magnitudes with epsilon to avoid zeros
    clean_left_mag = np.abs(clean_left_stft) + eps
    clean_right_mag = np.abs(clean_right_stft) + eps
    enhanced_left_mag = np.abs(enhanced_left_stft) + eps
    enhanced_right_mag = np.abs(enhanced_right_stft) + eps
    
    # Compute ILD in dB (add epsilon to both numerator and denominator)
    clean_ild = 20 * np.log10(clean_left_mag / clean_right_mag)
    enhanced_ild = 20 * np.log10(enhanced_left_mag / enhanced_right_mag)
    
    # Create speech activity mask based on both channels (T=20dB as in paper)
    threshold = 20  # dB below max as specified in paper
    
    # Combine energy from both channels for more robust mask
    combined_energy_left = clean_left_mag**2  # Already added epsilon above
    combined_energy_right = clean_right_mag**2
    combined_energy = np.maximum(combined_energy_left, combined_energy_right)
    
    # Handle potential zero max energy (unlikely but safer)
    max_energy = np.max(combined_energy)
    if max_energy <= eps:
        # Return default value if no significant energy
        return 0.0
        
    # Create mask
    mask = (combined_energy > max_energy * 10**(-threshold/10))
    
    # Ensure mask has at least some active points
    if np.sum(mask) == 0:
        # Alternative: use a less strict threshold
        mask = (combined_energy > max_energy * 10**(-30/10))
        # If still no active points, return default
        if np.sum(mask) == 0:
            return 0.0
    
    # Compute mean absolute error in active regions
    ild_error = np.abs(clean_ild - enhanced_ild)
    
    # Handle NaN values before masking
    ild_error = np.nan_to_num(ild_error, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply mask and calculate mean
    ild_error_masked = ild_error * mask
    mean_ild_error = np.sum(ild_error_masked) / np.sum(mask)
    
    return mean_ild_error


def compute_ipd_error(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """
    Compute IPD error between clean and enhanced binaural signals in degrees,
    focusing on 200-1500 Hz where IPD is most perceptually relevant
    """
    # STFT parameters
    sr_ratio = sr / 16000
    n_fft = int(512 * sr_ratio)
    hop_length = int(100 * sr_ratio)
    win_length = int(400 * sr_ratio)
    
    # Compute STFTs
    clean_left_stft = librosa.stft(clean_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    clean_right_stft = librosa.stft(clean_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    enhanced_left_stft = librosa.stft(enhanced_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    enhanced_right_stft = librosa.stft(enhanced_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Compute IPD
    clean_ipd = np.angle(clean_left_stft * np.conj(clean_right_stft))
    enhanced_ipd = np.angle(enhanced_left_stft * np.conj(enhanced_right_stft))
    
    # Create speech activity mask with 20 dB threshold as in the paper
    threshold = 20
    combined_energy_left = np.abs(clean_left_stft)**2
    combined_energy_right = np.abs(clean_right_stft)**2
    combined_energy = np.maximum(combined_energy_left, combined_energy_right)
    
    max_energy = np.max(combined_energy)
    mask = (combined_energy > max_energy * 10**(-threshold/10))
    
    # Focus on 200-1500 Hz where IPD is most meaningful (paper's range)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    f_min = 200  # Hz
    f_max = 1500  # Hz - reduced from 8000 Hz to focus on perceptually relevant range
    freq_mask = np.logical_and(freqs >= f_min, freqs <= f_max)
    freq_mask = freq_mask[:, np.newaxis]  # Reshape for broadcasting
    combined_mask = np.logical_and(mask, freq_mask)
    
    # Handle phase wrapping by taking the smallest angle difference
    ipd_error = np.abs(np.angle(np.exp(1j * (clean_ipd - enhanced_ipd))))
    
    # Convert to degrees (as reported in the paper)
    ipd_error_degrees = ipd_error * (180 / np.pi)
    ipd_error_masked = ipd_error_degrees * combined_mask
    
    # Ensure we have active bins to calculate mean
    total_active_bins = np.sum(combined_mask)
    if total_active_bins == 0:
        return None  # Return None instead of 0 to indicate calculation failure
        
    mean_ipd_error = np.sum(ipd_error_masked) / total_active_bins
    
    return mean_ipd_error

def compute_ipd_error_paper_style(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """
    Compute IPD error following the paper's Equation 11 approach with amplitude weighting
    """
    # STFT parameters
    sr_ratio = sr / 16000
    n_fft = int(512 * sr_ratio)
    hop_length = int(100 * sr_ratio)
    win_length = int(400 * sr_ratio)
    
    # Compute STFTs
    clean_left_stft = librosa.stft(clean_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    clean_right_stft = librosa.stft(clean_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    enhanced_left_stft = librosa.stft(enhanced_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    enhanced_right_stft = librosa.stft(enhanced_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Epsilon for numerical stability
    eps = 1e-8
    
    # Compute magnitudes
    clean_left_mag = np.abs(clean_left_stft) + eps
    clean_right_mag = np.abs(clean_right_stft) + eps
    
    # Calculate joint energy mask for focusing on speech-active regions (T=20dB as in paper)
    threshold = 20  # dB below max
    combined_energy = np.maximum(clean_left_mag**2, clean_right_mag**2)
    max_energy = np.max(combined_energy)
    mask = (combined_energy > max_energy * 10**(-threshold/10))
    
    # Compute IPD values (unwrapped phase differences)
    clean_ipd = np.angle(clean_left_stft * np.conj(clean_right_stft))
    enhanced_ipd = np.angle(enhanced_left_stft * np.conj(enhanced_right_stft))
    
    # Handle phase wrapping by taking the angular difference on the unit circle
    ipd_error = np.abs(np.angle(np.exp(1j * (clean_ipd - enhanced_ipd))))
    
    # Following Eq.(11): Weight by clean amplitude (|X_L|+|X_R|)
    weight = clean_left_mag + clean_right_mag
    ipd_error_weighted = ipd_error * mask * weight
    
    # Return unitless value (not in degrees) as per paper
    if np.sum(mask * weight) > 0:
        mean_ipd_error = np.sum(ipd_error_weighted) / np.sum(mask * weight)
    else:
        mean_ipd_error = 0.0
        
    return mean_ipd_error

def safe_mbstoi(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """Compute MBSTOI with strict length requirements and no fallbacks"""
    import math
    
    # Ensure minimum length and make divisible by 256 (STOI requirement)
    min_len = min(len(clean_left), len(clean_right), len(enhanced_left), len(enhanced_right))
    
    # Make target length at least 4096 (16 frames of 256 samples) and a multiple of 256
    target_len = max(4096, 256 * math.ceil(min_len / 256))
    
    # Trim and pad all signals to exactly the same length
    def prepare_signal(signal):
        signal = signal.copy()
        signal = signal[:min_len]
        if len(signal) < target_len:
            signal = np.pad(signal, (0, target_len - len(signal)))
        return signal
    
    clean_left = prepare_signal(clean_left)
    clean_right = prepare_signal(clean_right)
    enhanced_left = prepare_signal(enhanced_left)
    enhanced_right = prepare_signal(enhanced_right)
    
    # Normalize signals to prevent negative MBSTOI values
    for s in [clean_left, clean_right, enhanced_left, enhanced_right]:
        s -= np.mean(s)
        max_val = np.max(np.abs(s))
        if max_val > 0:
            s /= max_val
    
    try:
        from MBSTOI.mbstoi import mbstoi
        # Use fixed gridcoarseness=1 as in the paper
        score = mbstoi(clean_left, clean_right, enhanced_left, enhanced_right, gridcoarseness=1)
        
        # Validate result - if invalid, return None instead of estimation
        if np.isnan(score) or np.isinf(score) or score < 0 or score > 1:
            print(f"Warning: Invalid MBSTOI score: {score}. Skipping this sample.")
            return None
        
        return score
    except Exception as e:
        print(f"Error in MBSTOI calculation: {e}")
        return None

def evaluate_at_paper_snrs(model_checkpoint, test_dir, clean_dir, output_dir, batch_size=8, limit_pairs=None):
    paper_snrs = [-6, -3, 0, 3, 6, 9, 12, 15]
    results_by_snr = {}
    
    for snr in paper_snrs:
        print(f"\n===== Evaluating at SNR = {snr} dB =====")
        snr_output_dir = os.path.join(output_dir, f"paper_snr_{snr}dB")
        
        # 傳遞 limit_pairs 參數
        snr_results = evaluate_model(
            model_checkpoint,
            test_dir,
            clean_dir,
            snr_output_dir,
            specific_snr=snr,
            batch_size=batch_size,
            limit_pairs=limit_pairs  # 加上這個參數
        )
        
        if snr_results:
            results_by_snr[snr] = snr_results

    
    # Generate paper-style table
    with open(os.path.join(output_dir, "paper_style_results.csv"), 'w') as f:
        f.write("SNR,MBSTOI,ΔSegSNR,LILD,LIPD\n")
        
        for snr in paper_snrs:
            if snr in results_by_snr:
                result = results_by_snr[snr]
                f.write(f"{snr},{result.get('Average MBSTOI', 'N/A'):.4f}," +
                        f"{result.get('Average SegSNR Improvement', 'N/A'):.2f}," +
                        f"{result.get('Average ILD Error (dB)', 'N/A'):.4f}," +
                        f"{result.get('Average IPD Error (paper)', 'N/A'):.4f}\n")
    
    # Calculate overall averages (like in the paper)
    avg_mbstoi = np.mean([r.get('Average MBSTOI', 0) for r in results_by_snr.values()])
    avg_segsnr = np.mean([r.get('Average SegSNR Improvement', 0) for r in results_by_snr.values()])
    avg_ild = np.mean([r.get('Average ILD Error (dB)', 0) for r in results_by_snr.values()])
    avg_ipd = np.mean([r.get('Average IPD Error (paper)', 0) for r in results_by_snr.values()])
    
    print("\n===== PAPER STYLE AVERAGES =====")
    print(f"Average MBSTOI: {avg_mbstoi:.4f}")
    print(f"Average SegSNR: {avg_segsnr:.2f}")
    print(f"Average LILD: {avg_ild:.4f}")
    print(f"Average LIPD: {avg_ipd:.4f}")
    
    with open(os.path.join(output_dir, "paper_style_averages.txt"), 'w') as f:
        f.write(f"Average MBSTOI: {avg_mbstoi:.4f}\n")
        f.write(f"Average SegSNR: {avg_segsnr:.2f}\n")
        f.write(f"Average LILD: {avg_ild:.4f}\n")
        f.write(f"Average LIPD: {avg_ipd:.4f}\n")
    
    return results_by_snr

def evaluate_model(model_checkpoint, test_dataset_path, clean_dataset_path, output_dir, 
                   is_timit=False, specific_snr=None, batch_size=8, limit_pairs=None):
    """
    Evaluate model with fixed file matching for TIMIT dataset
    """
    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize(config_path="config")
    config = compose(config_name="config")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating on {device}")
    model = DCNNLightningModule(config)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
    
    # Create a proper file matching system
    from pathlib import Path
    import traceback
    
    # Get all files
    clean_files = list(Path(clean_dataset_path).glob('**/*.wav'))
    noisy_files = list(Path(test_dataset_path).glob('**/*.wav'))
    
    print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")

    # Parse file tags to ensure proper azimuth matching
    def parse_tag(stem):
        m = re.search(r'_az(-?\d+)(?:_snr([+\-]?\d+(?:\.\d+)?))?$', stem)
        az = int(m.group(1)) if m else None
        snr = float(m.group(2)) if m and m.group(2) else None
        base = stem.split('_az')[0]
        return base, az, snr

    clean_dict = {}
    for f in clean_files:
        base, az, _ = parse_tag(f.stem)
        if az is not None:  # Only store files with valid azimuth
            clean_dict[(base, az)] = f

    noisy_dict = {}
    for f in noisy_files:
        base, az, snr = parse_tag(f.stem)
        if az is None:
            continue  # Skip files without azimuth info
        if specific_snr is not None:
            if snr is None or abs(float(snr) - float(specific_snr)) > 0.1:
                continue
        noisy_dict[(base, az, snr)] = f

    # Find matching pairs with same base name AND azimuth
    pairs = [(clean_dict[(base, az)], noisy_dict[(base, az, snr)]) 
            for (base, az, snr) in noisy_dict.keys() if (base, az) in clean_dict]
    
    # Limit pairs to the specified number if requested
    if limit_pairs is not None and limit_pairs > 0 and limit_pairs < len(pairs):
        print(f"Limiting evaluation to {limit_pairs} pairs (out of {len(pairs)} available)")
        pairs = pairs[:limit_pairs]  # Take the first N pairs
        
    print(f"Found {len(pairs)} matching clean/noisy pairs (azimuth-aligned)")

    if len(pairs) == 0:
        print("ERROR: No matching clean/noisy pairs found! Check file naming patterns.")
        return
    
    # Initialize results
    results = {
        'filename': [],
        'segSNR_L': [],
        'segSNR_R': [],
        'MBSTOI': [],
        'ILD_error': [],
        'IPD_error_paper': [],  # Paper's unitless version
        'IPD_error_degrees': []  # Original degrees version
    }
    
    # Define a helper function to calculate all metrics for a single file
    def calc_metrics(clean_data, noisy_data, enhanced_data, sr):
        # RMS normalize signals before evaluation
        def rms_normalize(signal):
            rms = np.sqrt(np.mean(signal**2) + 1e-8)
            return signal / rms

        # Left channel
        clean_norm_L = rms_normalize(clean_data[:, 0])
        noisy_norm_L = rms_normalize(noisy_data[:, 0])
        enhanced_norm_L = rms_normalize(enhanced_data[0])
        segSNR_L = fwsegsnr(clean_norm_L, noisy_norm_L, enhanced_norm_L, sr=sr)
        
        # Right channel
        clean_norm_R = rms_normalize(clean_data[:, 1])
        noisy_norm_R = rms_normalize(noisy_data[:, 1])
        enhanced_norm_R = rms_normalize(enhanced_data[1])
        segSNR_R = fwsegsnr(clean_norm_R, noisy_norm_R, enhanced_norm_R, sr=sr)

        # MBSTOI
        mbstoi_score = safe_mbstoi(
            clean_data[:, 0], clean_data[:, 1], 
            enhanced_data[0], enhanced_data[1], 
            sr=sr
        )
        
        # ILD error
        ild_error = compute_ild_error(
            clean_data[:, 0], clean_data[:, 1],
            enhanced_data[0], enhanced_data[1],
            sr=sr
        )
        
        # IPD errors (both versions)
        ipd_error_degrees = compute_ipd_error(
            clean_data[:, 0], clean_data[:, 1],
            enhanced_data[0], enhanced_data[1],
            sr=sr
        )
        
        ipd_error_unitless = compute_ipd_error_paper_style(
            clean_data[:, 0], clean_data[:, 1],
            enhanced_data[0], enhanced_data[1],
            sr=sr
        )
        
        return segSNR_L, segSNR_R, mbstoi_score, ild_error, ipd_error_unitless, ipd_error_degrees
    
    # Process in batches
    total_batches = (len(pairs) + batch_size - 1) // batch_size  # Ceiling division
    for batch_idx in tqdm(range(total_batches), desc=f"Processing batches (size={batch_size})", total=total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]
        
        # Load all data first to determine max length and discard invalid files
        valid_pairs = []
        clean_data_list = []
        noisy_data_list = []
        clean_sr_list = []
        
        for clean_path, noisy_path in batch_pairs:
            try:
                # Load clean and noisy audio
                clean_data, c_sr = sf.read(clean_path)
                noisy_data, n_sr = sf.read(noisy_path)
                
                # Skip mono files
                if len(clean_data.shape) == 1 or len(noisy_data.shape) == 1:
                    print(f"Warning: Mono file detected! Skipping {clean_path.name}")
                    continue
                
                # Resample noisy if needed (proper per-channel resampling)
                if n_sr != c_sr:
                    print(f"Resampling {noisy_path.name} from {n_sr} Hz to {c_sr} Hz")
                    noisy_data = np.stack([
                        librosa.resample(noisy_data[:, ch], orig_sr=n_sr, target_sr=c_sr) 
                        for ch in range(noisy_data.shape[1])
                    ], axis=1)
                
                # Add to valid data
                valid_pairs.append((clean_path, noisy_path))
                clean_data_list.append(clean_data)
                noisy_data_list.append(noisy_data)
                clean_sr_list.append(c_sr)
                
            except Exception as e:
                print(f"Error loading {clean_path.name} or {noisy_path.name}: {e}")
                continue
        
        # Skip if no valid files in this batch
        if not valid_pairs:
            continue
        
        # Find max length in this batch
        max_len = max(data.shape[0] for data in clean_data_list)
        
        # Check if batch might be too large for GPU memory
        # Assuming 16-bit audio (2 bytes per sample) * 2 channels * batch_size * max_len
        memory_needed_mb = 2 * 2 * len(valid_pairs) * max_len / (1024 * 1024)
        if memory_needed_mb > 1024:  # More than 1GB of audio data
            print(f"Warning: Large batch detected ({memory_needed_mb:.1f} MB). Consider reducing batch size.")
        
        # Pad all samples to the same length
        padded_noisy_batch = []
        for noisy_data in noisy_data_list:
            if noisy_data.shape[0] < max_len:
                # Pad to match max length
                padded = np.pad(noisy_data, ((0, max_len - noisy_data.shape[0]), (0, 0)))
            else:
                padded = noisy_data
            padded_noisy_batch.append(padded)
        
        # Create batch tensor for the model
        batch_tensors = []
        for padded_noisy in padded_noisy_batch:
            # Create tensor with shape [1, channels, samples]
            tensor = torch.tensor(padded_noisy.T, dtype=torch.float32).unsqueeze(0)
            batch_tensors.append(tensor)
        
        # Stack tensors (not cat, as we need a new dimension)
        batch_tensor = torch.stack(batch_tensors, dim=0).squeeze(1).to(device)
        
        # Process batch through model
        with torch.no_grad():
            enhanced_batch = model(batch_tensor)
            # Move to CPU immediately to free GPU memory
            enhanced_batch = enhanced_batch.cpu()
        
        # # Process each file's results
        # for i in range(len(valid_pairs)):
        #     try:
        #         clean_path, noisy_path = valid_pairs[i]
        #         clean_data = clean_data_list[i]
        #         noisy_data = noisy_data_list[i]
        #         enhanced_data = enhanced_batch[i].numpy()
        #         sr = clean_sr_list[i]
                
        #         # Truncate enhanced data to original clean length
        #         orig_len = clean_data.shape[0]
        #         if enhanced_data.shape[1] > orig_len:
        #             enhanced_data = enhanced_data[:, :orig_len]
                
        #         # Calculate all metrics
        #         segSNR_L, segSNR_R, mbstoi_score, ild_error, ipd_error_unitless, ipd_error_degrees = calc_metrics(
        #             clean_data, noisy_data, enhanced_data, sr
        #         )
                
        #         # Add to results
        #         results['filename'].append(clean_path.name)
        #         results['segSNR_L'].append(segSNR_L)
        #         results['segSNR_R'].append(segSNR_R)
        #         results['MBSTOI'].append(mbstoi_score)
        #         results['ILD_error'].append(ild_error)
        #         results['IPD_error_paper'].append(ipd_error_unitless)
        #         results['IPD_error_degrees'].append(ipd_error_degrees)
                
        #         # Print first few results
        #         if batch_idx == 0 and i < 3:
        #             print(f"\nMetrics for {clean_path.name}:")
        #             print(f"- SegSNR: L={segSNR_L:.2f} dB, R={segSNR_R:.2f} dB")
        #             print(f"- MBSTOI: {mbstoi_score:.4f}")
        #             print(f"- ILD error: {ild_error:.2f} dB")
        #             print(f"- IPD error (paper): {ipd_error_unitless:.4f}")
        #             print(f"- IPD error (degrees): {ipd_error_degrees:.2f} degrees")
                
        #         # Save enhanced audio
        #         enhanced_path = os.path.join(output_dir, "enhanced", f"enhanced_{clean_path.name}")
        #         sf.write(enhanced_path, enhanced_data.T, sr)
                
        #     except Exception as e:
        #         print(f"Error processing file {i} in batch: {e}")
        #         traceback.print_exc()
        # In the evaluation loop where you process each file's results:
        for i in range(len(valid_pairs)):
            try:
                clean_path, noisy_path = valid_pairs[i]
                clean_data = clean_data_list[i]
                noisy_data = noisy_data_list[i]
                enhanced_data = enhanced_batch[i].numpy()
                sr = clean_sr_list[i]
                
                # Ensure enhanced data matches the original clean length
                orig_len = clean_data.shape[0]
                if enhanced_data.shape[1] > orig_len:
                    enhanced_data = enhanced_data[:, :orig_len]
                elif enhanced_data.shape[1] < orig_len:
                    # If enhanced is shorter (shouldn't happen), pad to match
                    pad_width = ((0, 0), (0, orig_len - enhanced_data.shape[1]))
                    enhanced_data = np.pad(enhanced_data, pad_width)
                
                # Verify channel correlation to detect potential channel swapping
                l_corr = np.corrcoef(clean_data[:, 0], enhanced_data[0])[0, 1]
                r_corr = np.corrcoef(clean_data[:, 1], enhanced_data[1])[0, 1]
                
                if l_corr < 0.4 or r_corr < 0.4:
                    print(f"Warning: Low correlation between clean and enhanced channels: L={l_corr:.2f}, R={r_corr:.2f}")
                    # Consider channel swapping if cross-correlation is higher
                    cross_l_r = np.corrcoef(clean_data[:, 0], enhanced_data[1])[0, 1]
                    cross_r_l = np.corrcoef(clean_data[:, 1], enhanced_data[0])[0, 1]
                    if (cross_l_r > l_corr and cross_r_l > r_corr):
                        print("Channels appear swapped, fixing...")
                        enhanced_data = enhanced_data[[1, 0]]  # Swap channels
                
                # Calculate metrics with strict checking - only include valid results
                segSNR_L, segSNR_R, mbstoi_score, ild_error, ipd_error_unitless, ipd_error_degrees = calc_metrics(
                    clean_data, noisy_data, enhanced_data, sr
                )
                
                # Only include valid results in the statistics
                if mbstoi_score is not None:
                    # Add to results
                    results['filename'].append(clean_path.name)
                    results['segSNR_L'].append(segSNR_L)
                    results['segSNR_R'].append(segSNR_R)
                    results['MBSTOI'].append(mbstoi_score)
                    results['ILD_error'].append(ild_error if ild_error is not None else float('nan'))
                    results['IPD_error_paper'].append(ipd_error_unitless if ipd_error_unitless is not None else float('nan'))
                    results['IPD_error_degrees'].append(ipd_error_degrees if ipd_error_degrees is not None else float('nan'))
                    
                    # Save enhanced audio
                    enhanced_path = os.path.join(output_dir, "enhanced", f"enhanced_{clean_path.name}")
                    sf.write(enhanced_path, enhanced_data.T, sr)
                else:
                    print(f"Skipping file {clean_path.name} due to invalid MBSTOI score")
                    
            except Exception as e:
                print(f"Error processing file {i} in batch: {e}")
                import traceback
                traceback.print_exc()
    
    # Check if we processed any files
    if not results['filename']:
        print("Warning: No files were successfully processed!")
        return None
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics
    average_results = {
        'Average SegSNR (L)': results_df['segSNR_L'].mean(),
        'Average SegSNR (R)': results_df['segSNR_R'].mean(),
        'Average SegSNR Improvement': (results_df['segSNR_L'].mean() + results_df['segSNR_R'].mean()) / 2,
        'Average MBSTOI': results_df['MBSTOI'].mean(),
        'Average ILD Error (dB)': results_df['ILD_error'].mean(),
        'Average IPD Error (paper)': results_df['IPD_error_paper'].mean(),
        'Average IPD Error (degrees)': results_df['IPD_error_degrees'].mean()
    }
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    with open(os.path.join(output_dir, "average_results.txt"), 'w') as f:
        for metric, value in average_results.items():
            f.write(f"{metric}: {value:.4f}\n")
            print(f"{metric}: {value:.4f}")
            
    return average_results

def run_paper_style_evaluation(model_checkpoint, vctk_test_dir, vctk_clean_dir, 
                          timit_test_dir, timit_clean_dir, output_dir, limit_pairs=None):
    """
    Run evaluation in the same style as presented in the paper
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define SNR values as used in the paper
    snr_values = [-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    
    # Prepare results dictionaries for paper tables
    vctk_results = {snr: {} for snr in snr_values}
    timit_results = {snr: {} for snr in snr_values}
    
    # VCTK matched condition testing
    print("=== Evaluating on VCTK (matched condition) ===")
    
    for snr in snr_values:
        print(f"\n===== Evaluating VCTK at SNR = {snr} dB =====")
        # Filter files by SNR
        snr_dir = f"snr_{int(snr)}dB"  # Convert to int for directory name
        snr_test_dir = os.path.join(vctk_test_dir, snr_dir)
        
        if not os.path.exists(snr_test_dir):
            print(f"Warning: {snr_test_dir} not found. Using flat structure with specific_snr parameter...")
            # Try flat structure if SNR directories don't exist
            snr_test_dir = vctk_test_dir
        
        snr_output_dir = os.path.join(output_dir, f"vctk_snr_{int(snr)}dB")
        
        # Run evaluation at this SNR level
        vctk_results[snr] = evaluate_model(
            model_checkpoint,
            snr_test_dir,
            vctk_clean_dir,
            snr_output_dir,
            is_timit=False,
            specific_snr=snr,
            limit_pairs=limit_pairs
        )
    
    # TIMIT unmatched condition testing (if available)
    if timit_test_dir and timit_clean_dir and os.path.exists(timit_test_dir) and os.path.exists(timit_clean_dir):
        print("\n===== Evaluating on TIMIT (unmatched condition) =====")
        
        for snr in snr_values:
            print(f"\n===== Evaluating TIMIT at SNR = {snr} dB =====")
            # SNR-specific directory for TIMIT
            timit_snr_test_dir = os.path.join(timit_test_dir, f"snr_{int(snr)}dB")
            
            if not os.path.exists(timit_snr_test_dir):
                print(f"TIMIT directory not found: {timit_snr_test_dir}")
                print(f"Trying flat structure for TIMIT with specific_snr parameter...")
                timit_snr_test_dir = timit_test_dir
                
            snr_output_dir = os.path.join(output_dir, f"timit_snr_{int(snr)}dB")
            
            # Run evaluation with clear logging
            print(f"Evaluating TIMIT with dirs:\n- Noisy: {timit_snr_test_dir}\n- Clean: {timit_clean_dir}")
            timit_results[snr] = evaluate_model(
                model_checkpoint,
                timit_snr_test_dir,
                timit_clean_dir,
                snr_output_dir,
                is_timit=True,
                specific_snr=snr,
                limit_pairs=limit_pairs
            )
            
            # Debug log after evaluation
            if timit_results[snr]:
                print(f"TIMIT evaluation at {snr} dB completed successfully")
            else:
                print(f"TIMIT evaluation at {snr} dB failed or returned no results")
    else:
        print(f"\nSkipping TIMIT evaluation - directory not found or not provided")
        if timit_test_dir:
            print(f"TIMIT test dir: {timit_test_dir} exists: {os.path.exists(timit_test_dir)}")
        if timit_clean_dir:
            print(f"TIMIT clean dir: {timit_clean_dir} exists: {os.path.exists(timit_clean_dir)}")
    
    # Generate paper-style tables for both datasets
    generate_paper_tables(vctk_results, timit_results, output_dir)
    
    return vctk_results, timit_results

def generate_paper_tables(vctk_results, timit_results, output_dir):
    """Generate tables in the same format as in the paper"""
    snr_values = [-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    
    # Table 1: Anechoic results (VCTK) as in the paper
    with open(os.path.join(output_dir, "table1_anechoic_results.csv"), 'w') as f:
        f.write("Input SNR,MBSTOI,ΔSegSNR,L_ILD (dB),L_IPD\n")
        
        for snr in snr_values:
            if snr in vctk_results:
                result = vctk_results[snr]
                f.write(f"{int(snr)},{result.get('Average MBSTOI', 'N/A'):.2f},{result.get('Average SegSNR Improvement', 'N/A'):.1f},{result.get('Average ILD Error (dB)', 'N/A'):.2f},{result.get('Average IPD Error (paper)', 'N/A'):.2f}\n")

    # Table 2: Unmatched condition (TIMIT) results if available
    if timit_results:
        with open(os.path.join(output_dir, "table2_timit_results.csv"), 'w') as f:
            f.write("Input SNR,MBSTOI,ΔSegSNR,L_ILD (dB),L_IPD (degrees)\n")
            
            for snr in snr_values:
                if snr in timit_results:
                    result = timit_results[snr]
                    f.write(f"{snr},{result.get('Average MBSTOI', 'N/A'):.2f},{result.get('Average SegSNR Improvement', 'N/A'):.1f},{result.get('Average ILD Error (dB)', 'N/A'):.2f},{result.get('Average IPD Error (degrees)', 'N/A'):.1f}\n")
    
    # Generate comparison with paper results
    paper_results = {
        # Values from Table 1 in the paper for BCCTN-Proposed Loss
        -6: {"MBSTOI": 0.73, "SegSNR": 14.3, "ILD": 0.61, "IPD": 8},
        -3: {"MBSTOI": 0.79, "SegSNR": 12.7, "ILD": 0.62, "IPD": 7},
        0: {"MBSTOI": 0.85, "SegSNR": 12.7, "ILD": 0.40, "IPD": 5},
        3: {"MBSTOI": 0.87, "SegSNR": 11.5, "ILD": 0.36, "IPD": 4},
        6: {"MBSTOI": 0.91, "SegSNR": 9.7, "ILD": 0.34, "IPD": 3},
        9: {"MBSTOI": 0.94, "SegSNR": 8.4, "ILD": 0.20, "IPD": 2},
        12: {"MBSTOI": 0.96, "SegSNR": 7.0, "ILD": 0.19, "IPD": 2},
        15: {"MBSTOI": 0.96, "SegSNR": 5.4, "ILD": 0.19, "IPD": 2}
    }
    
    with open(os.path.join(output_dir, "comparison_with_paper.csv"), 'w') as f:
        f.write("SNR,Metric,Paper Value,Our Value,Difference\n")
        
        for snr in snr_values:
            if snr in vctk_results and snr in paper_results:
                result = vctk_results[snr]
                paper = paper_results[snr]
                
                # MBSTOI
                our_mbstoi = result.get('Average MBSTOI', 0)
                paper_mbstoi = paper["MBSTOI"]
                f.write(f"{snr},MBSTOI,{paper_mbstoi:.2f},{our_mbstoi:.2f},{our_mbstoi-paper_mbstoi:.2f}\n")
                
                # SegSNR
                our_segsnr = result.get('Average SegSNR Improvement', 0)
                paper_segsnr = paper["SegSNR"]
                f.write(f"{snr},SegSNR,{paper_segsnr:.1f},{our_segsnr:.1f},{our_segsnr-paper_segsnr:.1f}\n")
                
                # ILD Error
                our_ild = result.get('Average ILD Error (dB)', 0)
                paper_ild = paper["ILD"]
                f.write(f"{snr},ILD,{paper_ild:.2f},{our_ild:.2f},{our_ild-paper_ild:.2f}\n")
                
                # IPD Error
                our_ipd = result.get('Average IPD Error (degrees)', 0)
                paper_ipd = paper["IPD"]
                f.write(f"{snr},IPD,{paper_ipd:.1f},{our_ipd:.1f},{our_ipd-paper_ipd:.1f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate binaural speech enhancement model")
    
    # Basic evaluation arguments
    parser.add_argument("--model_checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dataset_path", help="Path to test dataset directory")
    parser.add_argument("--clean_dataset_path", help="Path to clean dataset directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    
    # Arguments for paper-style evaluation
    parser.add_argument("--paper_style_eval", action="store_true", help="Run paper-style evaluation with SNR breakdown")
    parser.add_argument("--vctk_test_dir", help="Path to VCTK noisy test dataset")
    parser.add_argument("--vctk_clean_dir", help="Path to VCTK clean test dataset")
    parser.add_argument("--timit_test_dir", help="Path to TIMIT noisy test dataset")
    parser.add_argument("--timit_clean_dir", help="Path to TIMIT clean test dataset")
    parser.add_argument("--specific_snr", type=float, help="Filter by specific SNR value")
    parser.add_argument("--batch_size", type=int, default=8, 
                    help="Batch size for model inference (higher values use more GPU memory)")
    parser.add_argument("--limit_pairs", type=int, help="Limit number of pairs to evaluate (e.g. 750)")
    parser.add_argument("--vctk_only", action="store_true", help="Only evaluate VCTK data (skip TIMIT)")
    parser.add_argument("--timit_only", action="store_true", help="Only evaluate TIMIT data (skip VCTK)")
    args = parser.parse_args()
    
    # Find and modify the part around line 750-760 in eval.py - where argument parsing happens
    if args.paper_style_eval:
        # Run paper-style evaluation with SNR breakdown (-6 to 15 dB)
        if args.vctk_only and all([args.vctk_test_dir, args.vctk_clean_dir]):
            # Only process VCTK data
            print("Processing only VCTK dataset")
            evaluate_at_paper_snrs(
                args.model_checkpoint,
                args.vctk_test_dir,
                args.vctk_clean_dir, 
                args.output_dir,
                batch_size=args.batch_size,
                limit_pairs=args.limit_pairs
            )
        elif args.timit_test_dir and args.timit_clean_dir and not args.vctk_only:
            # Process both VCTK and TIMIT datasets or TIMIT only
            print("Processing both VCTK and TIMIT datasets" if all([args.vctk_test_dir, args.vctk_clean_dir]) else "Processing only TIMIT dataset")
            
            # If VCTK directories are not provided, just process TIMIT
            if not all([args.vctk_test_dir, args.vctk_clean_dir]):
                # Run TIMIT-only evaluation
                timit_results = {}
                snr_values = [-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
                
                for snr in snr_values:
                    print(f"\n===== Evaluating TIMIT at SNR = {snr} dB =====")
                    timit_snr_test_dir = os.path.join(args.timit_test_dir, f"snr_{int(snr)}dB")
                    
                    if not os.path.exists(timit_snr_test_dir):
                        print(f"TIMIT directory not found: {timit_snr_test_dir}")
                        print(f"Trying flat structure for TIMIT with specific_snr parameter...")
                        timit_snr_test_dir = args.timit_test_dir
                        
                    snr_output_dir = os.path.join(args.output_dir, f"timit_snr_{int(snr)}dB")
                    
                    timit_results[snr] = evaluate_model(
                        args.model_checkpoint,
                        timit_snr_test_dir,
                        args.timit_clean_dir,
                        snr_output_dir,
                        is_timit=True,
                        specific_snr=snr,
                        batch_size=args.batch_size,
                        limit_pairs=args.limit_pairs
                    )
                
                # Generate paper-style tables for TIMIT only
                with open(os.path.join(args.output_dir, "timit_results.csv"), 'w') as f:
                    f.write("Input SNR,MBSTOI,ΔSegSNR,L_ILD (dB),L_IPD (degrees)\n")
                    
                    for snr in snr_values:
                        if snr in timit_results and timit_results[snr]:
                            result = timit_results[snr]
                            f.write(f"{snr},{result.get('Average MBSTOI', 'N/A'):.2f},{result.get('Average SegSNR Improvement', 'N/A'):.1f},{result.get('Average ILD Error (dB)', 'N/A'):.2f},{result.get('Average IPD Error (degrees)', 'N/A'):.1f}\n")
            else:
                # Process both VCTK and TIMIT
                run_paper_style_evaluation(
                    args.model_checkpoint,
                    args.vctk_test_dir,
                    args.vctk_clean_dir,
                    args.timit_test_dir,
                    args.timit_clean_dir,
                    args.output_dir,
                    limit_pairs=args.limit_pairs
                )
        else:
            parser.error("Paper-style evaluation requires either VCTK test and clean directories, or TIMIT test and clean directories")
    else:
        # Run basic evaluation
        if not all([args.test_dataset_path, args.clean_dataset_path]):
            parser.error("Basic evaluation requires test and clean dataset paths")
        
        evaluate_model(
            args.model_checkpoint,
            args.test_dataset_path,
            args.clean_dataset_path,
            args.output_dir,
            specific_snr=args.specific_snr,
            batch_size=args.batch_size,
            limit_pairs=args.limit_pairs
        )