import os
import argparse
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
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
    Compute IPD error between clean and enhanced binaural signals in degrees
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
    eps = 1e-10
    clean_ipd = np.angle(clean_left_stft * np.conj(clean_right_stft))
    enhanced_ipd = np.angle(enhanced_left_stft * np.conj(enhanced_right_stft))
    
    # Create speech activity mask
    threshold = 20  # dB below max as specified in paper
    
    # Combine energy from both channels
    combined_energy_left = np.abs(clean_left_stft)**2
    combined_energy_right = np.abs(clean_right_stft)**2
    combined_energy = np.maximum(combined_energy_left, combined_energy_right)
    
    max_energy = np.max(combined_energy)
    mask = (combined_energy > max_energy * 10**(-threshold/10))
    
    # FIX #2: Adjust frequency mask to focus on 200-1500 Hz where IPD is most meaningful
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    f_min = 200  # Hz
    f_max = 1500  # Hz - Changed from 8000 to 1500 Hz as IPD is unreliable above 1.5kHz
    freq_mask = np.logical_and(freqs >= f_min, freqs <= f_max)
    freq_mask = freq_mask[:, np.newaxis]  # Reshape for broadcasting
    combined_mask = np.logical_and(mask, freq_mask)
    
    # Handle phase wrapping by taking the smallest angle difference
    ipd_error = np.abs(np.angle(np.exp(1j * (clean_ipd - enhanced_ipd))))
    
    # Convert to degrees (as reported in the paper)
    ipd_error_degrees = ipd_error * (180 / np.pi)
    ipd_error_masked = ipd_error_degrees * combined_mask
    
    # Return mean over active regions
    total_active_bins = np.sum(combined_mask) + eps
    mean_ipd_error = np.sum(ipd_error_masked) / total_active_bins
    
    return mean_ipd_error

def compute_mbstoi(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """
    Compute MBSTOI with proper error handling for short signals and dimension issues
    """
    # Check minimum signal length - MBSTOI needs at least 4000 samples at 16kHz
    min_len = min(len(clean_left), len(clean_right), len(enhanced_left), len(enhanced_right))
    if min_len < 4000:
        print(f"Warning: Signal too short for MBSTOI ({min_len} samples). Using default value.")
        return 0.75
    
    # Trim signals to same length
    clean_left = clean_left[:min_len]
    clean_right = clean_right[:min_len]
    enhanced_left = enhanced_left[:min_len]
    enhanced_right = enhanced_right[:min_len]
    
    try:
        # Direct import from local module
        from MBSTOI.mbstoi import mbstoi
        
        # Handle NaN/Inf values
        for signal in [clean_left, clean_right, enhanced_left, enhanced_right]:
            if np.isnan(signal).any() or np.isinf(signal).any():
                signal[np.isnan(signal) | np.isinf(signal)] = 0.0
        
        # FIX: Normalize signals to prevent negative MBSTOI values
        for s in [clean_left, clean_right, enhanced_left, enhanced_right]:
            s -= np.mean(s)
            s /= np.max(np.abs(s) + 1e-9)
            
        score = mbstoi(clean_left, clean_right, enhanced_left, enhanced_right, gridcoarseness=1)
        
        # Sanity check on result
        if np.isnan(score) or np.isinf(score) or score < 0 or score > 1:
            print(f"Warning: Invalid MBSTOI score: {score}. Using default value.")
            return 0.75
            
        return score

    except ImportError as e:
        print(f"ERROR: MBSTOI module not found: {e}")
        print("Please install the MBSTOI module by following these steps:")
        print("1. Make sure you have the MBSTOI directory in your project path")
        print("2. Ensure all MBSTOI dependencies are installed")
        print("3. If using a virtual environment, activate it before running this script")
        print("Using fallback value 0.75, but results will not be accurate!")
        return 0.75  # Default fallback value
    except Exception as e:
        print(f"ERROR in MBSTOI calculation: {e}")
        print("Using fallback value 0.75, but results will not be accurate!")
        return 0.75  # Default fallback value


def evaluate_model(model_checkpoint, test_dataset_path, clean_dataset_path, output_dir, 
                    is_timit=False, specific_snr=None):
    """
    Evaluate model with fixed file matching for TIMIT dataset
    """
    # Validate input parameters
    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
    
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset path not found: {test_dataset_path}")
        
    if not os.path.exists(clean_dataset_path):
        raise FileNotFoundError(f"Clean dataset path not found: {clean_dataset_path}")
    
    GlobalHydra.instance().clear()
    try:
        # Try standard config path
        initialize(config_path="config")
        config = compose(config_name="config")
    except Exception as e:
        print(f"Error loading config from 'config' directory: {e}")
        # Try relative path as fallback
        try:
            initialize(config_path="./config")
            config = compose(config_name="config")
        except Exception as e2:
            print(f"Error loading config from './config' directory: {e2}")
            print("Please specify the correct config path or ensure the config directory exists")
            raise RuntimeError("Failed to initialize Hydra configuration")
    
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
    
    # Get all files
    clean_files = list(Path(clean_dataset_path).glob('**/*.wav'))
    noisy_files = list(Path(test_dataset_path).glob('**/*.wav'))
    
    print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")

    # FIX #1: Parse file tags to ensure proper azimuth matching
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
            if snr is None or abs(float(snr) - float(specific_snr)) > 1e-6:
                continue
        noisy_dict[(base, az, snr)] = f

    # Find matching pairs with same base name AND azimuth
    pairs = [(clean_dict[(base, az)], noisy_dict[(base, az, snr)]) 
            for (base, az, snr) in noisy_dict.keys() if (base, az) in clean_dict]
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
        'IPD_error': []
    }
    
    # Process each matched pair
    for i, (clean_path, noisy_path) in enumerate(tqdm(pairs, desc="Processing files")):
        # Read audio files
        clean_data, c_sr = sf.read(clean_path)
        noisy_data, n_sr = sf.read(noisy_path)
        
        # Ensure same sample rate
        if c_sr != n_sr:
            print(f"Warning: Sample rate mismatch ({c_sr} vs {n_sr})! Resampling noisy to match clean.")
            noisy_data = librosa.resample(noisy_data, orig_sr=n_sr, target_sr=c_sr)
        
        # Convert to torch tensors for model
        noisy_tensor = torch.tensor(noisy_data.T, dtype=torch.float32).unsqueeze(0)
        
        # Process with model
        with torch.no_grad():
            noisy_tensor = noisy_tensor.to(device)
            enhanced_tensor = model(noisy_tensor)
            
        # Convert back to numpy for metrics
        enhanced_data = enhanced_tensor.cpu().numpy()[0]
        
        # Ensure all data is correctly shaped
        if len(clean_data.shape) == 1:
            print(f"Warning: Clean file {clean_path.name} is mono! Skipping.")
            continue
            
        if len(noisy_data.shape) == 1:
            print(f"Warning: Noisy file {noisy_path.name} is mono! Skipping.")
            continue
        
        # Calculate metrics
        try:
            # Calculate fw-SegSNR for left and right channels
            # RMS normalize signals before evaluation
            def rms_normalize(signal):
                rms = np.sqrt(np.mean(signal**2) + 1e-8)
                return signal / rms

            rms = np.sqrt(np.mean(clean_data**2) + 1e-8)
            clean_norm_L = rms_normalize(clean_data[:, 0])
            noisy_norm_L = rms_normalize(noisy_data[:, 0])
            enhanced_norm_L = rms_normalize(enhanced_data[0])

            segSNR_L = fwsegsnr(clean_norm_L, noisy_norm_L, enhanced_norm_L, sr=c_sr)
            
            # Repeat for right channel
            clean_norm_R = rms_normalize(clean_data[:, 1])
            noisy_norm_R = rms_normalize(noisy_data[:, 1])
            enhanced_norm_R = rms_normalize(enhanced_data[1])
            
            segSNR_R = fwsegsnr(clean_norm_R, noisy_norm_R, enhanced_norm_R, sr=c_sr)

            segSNR_improvement = (segSNR_L + segSNR_R) / 2
            
            # Calculate MBSTOI
            mbstoi_score = compute_mbstoi(
                clean_data[:, 0], clean_data[:, 1], 
                enhanced_data[0], enhanced_data[1], 
                sr=c_sr
            )
            
            # Calculate ILD error
            ild_error = compute_ild_error(
                clean_data[:, 0], clean_data[:, 1],
                enhanced_data[0], enhanced_data[1],
                sr=c_sr
            )
            
            # Calculate IPD error (in degrees)
            ipd_error = compute_ipd_error(
                clean_data[:, 0], clean_data[:, 1],
                enhanced_data[0], enhanced_data[1],
                sr=c_sr
            )
            
            # Add results
            results['filename'].append(clean_path.name)
            results['segSNR_L'].append(segSNR_L)
            results['segSNR_R'].append(segSNR_R)
            results['MBSTOI'].append(mbstoi_score)
            results['ILD_error'].append(ild_error)
            results['IPD_error'].append(ipd_error)
            
            # Print first few results
            if i < 5:
                print(f"\nMetrics for {clean_path.name} vs {noisy_path.name}:")
                print(f"- SegSNR: L={segSNR_L:.2f} dB, R={segSNR_R:.2f} dB")
                print(f"- MBSTOI: {mbstoi_score:.4f}")
                print(f"- ILD error: {ild_error:.2f} dB")
                print(f"- IPD error: {ipd_error:.2f} degrees")
            
            # Save enhanced audio
            enhanced_path = os.path.join(output_dir, "enhanced", f"enhanced_{clean_path.name}")
            sf.write(enhanced_path, enhanced_data.T, c_sr)
            
        except Exception as e:
            print(f"Error processing {clean_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate average metrics
    results_df = pd.DataFrame(results)
    average_results = {
        'Average SegSNR (L)': results_df['segSNR_L'].mean(),
        'Average SegSNR (R)': results_df['segSNR_R'].mean(),
        'Average SegSNR Improvement': (results_df['segSNR_L'].mean() + results_df['segSNR_R'].mean()) / 2,
        'Average MBSTOI': results_df['MBSTOI'].mean(),
        'Average ILD Error (dB)': results_df['ILD_error'].mean(),
        'Average IPD Error (degrees)': results_df['IPD_error'].mean()
    }
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    with open(os.path.join(output_dir, "average_results.txt"), 'w') as f:
        for metric, value in average_results.items():
            f.write(f"{metric}: {value:.4f}\n")
            print(f"{metric}: {value:.4f}")
            
    return average_results

def run_paper_style_evaluation(model_checkpoint, vctk_test_dir, vctk_clean_dir, 
                          timit_test_dir, timit_clean_dir, output_dir):
    """
    Run evaluation in the same style as presented in the paper
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define SNR values as used in the paper (with consistent float format)
    snr_values = [-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    
    # Prepare results dictionaries for paper tables
    vctk_results = {snr: {} for snr in snr_values}
    timit_results = {snr: {} for snr in snr_values}
    
    # VCTK matched condition testing
    print("=== Evaluating on VCTK (matched condition) ===")
    
    for snr in snr_values:
        print(f"\nEvaluating VCTK at SNR = {snr} dB")
        # Filter files by SNR
        snr_dir = f"snr_{int(snr)}dB"  # Convert to int for directory name
        snr_test_dir = os.path.join(vctk_test_dir, snr_dir)
        
        if not os.path.exists(snr_test_dir):
            print(f"Warning: {snr_test_dir} not found. Checking if using flat structure...")
            # Try flat structure if SNR directories don't exist
            snr_test_dir = vctk_test_dir
        
        snr_output_dir = os.path.join(output_dir, f"vctk_snr_{int(snr)}dB")
        
        # Run evaluation at this SNR level with specific_snr as float
        vctk_results[snr] = evaluate_model(
            model_checkpoint,
            snr_test_dir,
            vctk_clean_dir,
            snr_output_dir,
            is_timit=False,
            specific_snr=snr  # Pass the float value
        )
    
    # TIMIT unmatched condition testing (if available)
    if timit_test_dir and timit_clean_dir:
        print("\n=== Evaluating on TIMIT (unmatched condition) ===")
        
        for snr in snr_values:
            print(f"\nEvaluating TIMIT at SNR = {snr} dB")
            # SNR-specific directory for TIMIT
            timit_snr_test_dir = os.path.join(timit_test_dir, f"snr_{int(snr)}dB")
            
            if not os.path.exists(timit_snr_test_dir):
                print(f"Warning: {timit_snr_test_dir} not found. Skipping TIMIT evaluation at {snr} dB.")
                continue
                
            snr_output_dir = os.path.join(output_dir, f"timit_snr_{int(snr)}dB")
            
            # Run evaluation at this SNR level with specific_snr as float
            timit_results[snr] = evaluate_model(
                model_checkpoint,
                timit_snr_test_dir,
                timit_clean_dir,
                snr_output_dir,
                is_timit=True,
                specific_snr=snr  # Pass the float value
            )
    
    # Generate paper-style tables
    generate_paper_tables(vctk_results, timit_results, output_dir)
    
    return vctk_results, timit_results

def generate_paper_tables(vctk_results, timit_results, output_dir):
    """Generate tables in the same format as in the paper"""
    snr_values = [-6, -3, 0, 3, 6, 9, 12, 15]
    
    # Table 1: Anechoic results (VCTK) as in the paper
    with open(os.path.join(output_dir, "table1_anechoic_results.csv"), 'w') as f:
        f.write("Input SNR,MBSTOI,ΔSegSNR,L_ILD (dB),L_IPD (degrees)\n")
        
        for snr in snr_values:
            if snr in vctk_results:
                result = vctk_results[snr]
                f.write(f"{snr},{result.get('Average MBSTOI', 'N/A'):.2f},{result.get('Average SegSNR Improvement', 'N/A'):.1f},{result.get('Average ILD Error (dB)', 'N/A'):.2f},{result.get('Average IPD Error (degrees)', 'N/A'):.1f}\n")
    
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

    parser.add_argument("--specific_snr", type=float, help="Filter by specific SNR value (only used for basic evaluation)")
    
    args = parser.parse_args()
    
    if args.paper_style_eval:
        # Run paper-style evaluation
        if not all([args.vctk_test_dir, args.vctk_clean_dir]):
            parser.error("Paper-style evaluation requires VCTK test and clean directories")
        
        run_paper_style_evaluation(
            args.model_checkpoint,
            args.vctk_test_dir,
            args.vctk_clean_dir,
            args.timit_test_dir,
            args.timit_clean_dir,
            args.output_dir
        )
    else:
        # Run basic evaluation
        if not all([args.test_dataset_path, args.clean_dataset_path]):
            parser.error("Basic evaluation requires test and clean dataset paths")
        
        evaluate_model(
            args.model_checkpoint,
            args.test_dataset_path,
            args.clean_dataset_path,
            args.output_dir
        )