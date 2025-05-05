"""
Fixes for Binaural Speech Enhancement Evaluation
================================================

This module contains updated functions to fix the evaluation issues
and make results comparable to the paper:
"Binaural Speech Enhancement using Deep Complex Convolutional Transformer Networks"
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
import re
import os
from tqdm import tqdm


# 1. STFT parameters - ensure consistency throughout
SAMPLE_RATE = 16000
FFT_SIZE = 512
WIN_LENGTH = 400  # 25ms at 16kHz
HOP_LENGTH = 100  # 6.25ms at 16kHz


# 2. Signal preparation for evaluation
def prepare_signals(clean, enhanced, target_sr=SAMPLE_RATE):
    """
    Prepare clean and enhanced signals to have the same length and sampling rate
    
    Args:
        clean: Clean reference signal (numpy array)
        enhanced: Enhanced signal (numpy array)
        target_sr: Target sampling rate
        
    Returns:
        Prepared clean and enhanced signals
    """
    # Ensure same length
    min_len = min(len(clean), len(enhanced))
    if len(clean) > min_len:
        clean = clean[:min_len]
    elif len(enhanced) > min_len:
        enhanced = enhanced[:min_len]
    
    # Ensure minimum length for STFT processing
    min_required = WIN_LENGTH + HOP_LENGTH * 10  # At least 10 frames
    if min_len < min_required:
        pad_length = min_required - min_len
        clean = np.pad(clean, (0, pad_length))
        enhanced = np.pad(enhanced, (0, pad_length))
    
    # Energy normalization for fair comparison
    clean = clean / (np.sqrt(np.mean(clean**2)) + 1e-8)
    enhanced = enhanced / (np.sqrt(np.mean(enhanced**2)) + 1e-8)
    
    return clean, enhanced


# 3. Updated frequency-weighted Segmental SNR
def fwsegsnr(clean, noisy, enhanced, sr=SAMPLE_RATE):
    """
    Compute frequency-weighted Segmental SNR improvement following VOICEBOX implementation
    
    Args:
        clean: Clean reference signal
        noisy: Noisy signal
        enhanced: Enhanced signal
        sr: Sample rate
        
    Returns:
        SNR improvement in dB (positive means enhancement is better than noisy)
    """
    # Frame parameters
    frame = WIN_LENGTH
    hop = HOP_LENGTH
    fft_size = FFT_SIZE
    
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
    W_valid = W[valid] / np.sum(W[valid])                  # Renormalize weights
    
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
    E_frames = _frames(enhanced - clean)    # Enhanced error component
    
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
    
    # Calculate SNR per band in dB, then apply weights
    def _band_snr(P_sig, P_noise):
        snr_band = 10 * np.log10(P_sig / P_noise)
        snr_band = np.clip(snr_band, -10, 35)  # Clip to [-10, 35] dB range
        return np.sum(snr_band * W_valid[None, :], axis=1) / np.sum(W_valid)
    
    # Calculate frame energies to identify active speech frames
    frame_energy = 10 * np.log10(np.sum(P_clean, axis=1) + 1e-10)
    active_frames = frame_energy > (np.max(frame_energy) - 40)  # Frames within 40dB of max
    
    # Calculate SNR for each frame, only for active frames
    if not np.any(active_frames):
        return 0.0  # No active frames
        
    snr_noisy = _band_snr(P_clean, P_noisy)[active_frames]
    snr_enh = _band_snr(P_clean, P_enh)[active_frames]
    
    # Return improvement
    return np.mean(snr_enh) - np.mean(snr_noisy)


def get_absolute_segsnr(clean, signal, sr=SAMPLE_RATE):
    """
    Compute absolute frequency-weighted Segmental SNR for any signal
    
    Args:
        clean: Clean reference signal
        signal: Signal to evaluate (noisy or enhanced)
        sr: Sample rate
        
    Returns:
        Absolute SNR in dB
    """
    # Frame parameters
    frame = WIN_LENGTH
    hop = HOP_LENGTH
    fft_size = FFT_SIZE
    
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
    W_valid = W[valid] / np.sum(W[valid])                  # Renormalize weights
    
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
    S_frames = _frames(signal - clean)  # Signal error component
    
    # Apply FFT
    C_fft = np.fft.rfft(C_frames, fft_size, axis=1)
    S_fft = np.fft.rfft(S_frames, fft_size, axis=1)
    
    # Compute band powers for each frame
    def _band_pow(X_fft):
        P = np.zeros((X_fft.shape[0], len(valid)))
        for b in range(len(valid)):
            lo, hi = edges_bin[b], edges_bin[b+1]
            P[:, b] = np.sum(np.abs(X_fft[:, lo:hi])**2, axis=1)
        return P + 1e-10  # Avoid division by zero
    
    # Calculate powers in each band
    P_clean = _band_pow(C_fft)
    P_signal = _band_pow(S_fft)
    
    # Calculate SNR per band in dB, then apply weights
    def _band_snr(P_sig, P_noise):
        snr_band = 10 * np.log10(P_sig / P_noise)
        snr_band = np.clip(snr_band, -10, 35)  # Clip to [-10, 35] dB range
        return np.sum(snr_band * W_valid[None, :], axis=1) / np.sum(W_valid)
    
    # Calculate frame energies to identify active speech frames
    frame_energy = 10 * np.log10(np.sum(P_clean, axis=1) + 1e-10)
    active_frames = frame_energy > (np.max(frame_energy) - 40)  # Frames within 40dB of max
    
    # Calculate SNR for each frame, only for active frames
    if not np.any(active_frames):
        return 0.0  # No active frames
        
    snr_signal = _band_snr(P_clean, P_signal)[active_frames]
    
    # Return absolute SNR
    return np.mean(snr_signal)


# 4. Updated ILD error calculation with speech activity mask
def compute_ild_error_with_mask(clean_left, clean_right, enhanced_left, enhanced_right, sr=SAMPLE_RATE):
    """
    Compute ILD error between clean and enhanced binaural signals with
    proper speech activity masking as described in the paper
    
    Args:
        clean_left, clean_right: Clean reference signals
        enhanced_left, enhanced_right: Enhanced signals
        sr: Sample rate
        
    Returns:
        ILD error in dB
    """
    # STFT parameters
    n_fft = FFT_SIZE
    hop_length = HOP_LENGTH
    win_length = WIN_LENGTH
    
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
    
    # Compute ILD in dB
    clean_ild = 20 * np.log10(clean_left_mag / clean_right_mag)
    enhanced_ild = 20 * np.log10(enhanced_left_mag / enhanced_right_mag)
    
    # Create speech activity mask as in the paper (T=20dB)
    threshold = 20  # dB below max as specified in paper
    
    # Combine energy from both channels for more robust mask
    combined_energy = np.maximum(clean_left_mag**2, clean_right_mag**2)
    max_energy = np.max(combined_energy)
    
    # Create Ideal Binary Mask (IBM) as described in the paper's Eq. (13-14)
    # For each frequency bin, threshold relative to maximum energy in that frequency
    mask = np.zeros_like(combined_energy, dtype=bool)
    for k in range(combined_energy.shape[0]):
        max_energy_k = np.max(combined_energy[k])
        mask[k] = 20 * np.log10(combined_energy[k]) > (20 * np.log10(max_energy_k) - threshold)
    
    # Compute mean absolute error in active regions
    ild_error = np.abs(clean_ild - enhanced_ild)
    
    # Handle NaN values before masking
    ild_error = np.nan_to_num(ild_error, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply mask and calculate mean
    active_bins = np.sum(mask)
    if active_bins == 0:
        return 0.0  # Return 0 if no active bins
        
    ild_error_masked = ild_error * mask
    mean_ild_error = np.sum(ild_error_masked) / active_bins
    
    return mean_ild_error


# 5. Updated IPD error calculation with speech activity mask
def compute_ipd_error_with_mask(
        clean_left, clean_right, enhanced_left, enhanced_right,
        sr=16000, f_max=1500):
    n_fft, hop, win = 512, 100, 400
    cl = librosa.stft(clean_left, n_fft=n_fft, hop_length=hop, win_length=win)
    cr = librosa.stft(clean_right, n_fft=n_fft, hop_length=hop, win_length=win)
    el = librosa.stft(enhanced_left, n_fft=n_fft, hop_length=hop, win_length=win)
    er = librosa.stft(enhanced_right, n_fft=n_fft, hop_length=hop, win_length=win)

    eps = 1e-8
    mag_cl = np.abs(cl) + eps
    mag_cr = np.abs(cr) + eps
    energy = np.maximum(mag_cl**2, mag_cr**2)

    # Speech‑active IBM (T = 20 dB)
    max_per_band = energy.max(axis=1, keepdims=True)
    mask = 20*np.log10(energy) > 20*np.log10(max_per_band) - 20

    # Limit to ≤ f_max
    max_bin = int(f_max / (sr / n_fft))
    mask = mask[:max_bin+1]
    ipd_clean = np.angle(cl * np.conj(cr))[:max_bin+1]
    ipd_enh   = np.angle(el * np.conj(er))[:max_bin+1]

    ipd_err = np.abs(np.angle(np.exp(1j*(ipd_clean - ipd_enh))))
    if not mask.any():
        return 0.0
    return (ipd_err[mask]).mean()


# 6. Safe MBSTOI calculation
def safe_mbstoi(clean_left, clean_right, enhanced_left, enhanced_right, sr=SAMPLE_RATE):
    """
    Compute MBSTOI with proper error handling and signal preparation
    
    Args:
        clean_left, clean_right: Clean reference signals
        enhanced_left, enhanced_right: Enhanced signals
        sr: Sample rate
        
    Returns:
        MBSTOI score or None if calculation fails
    """
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
        
        # Validate result
        if np.isnan(score) or np.isinf(score) or score < 0 or score > 1:
            print(f"Warning: Invalid MBSTOI score: {score}. Returning None.")
            return None
        
        return score
    except Exception as e:
        print(f"Error in MBSTOI calculation: {e}")
        return None


# 7. Safe PESQ calculation for binaural signals
def safe_pesq(clean_left, clean_right, processed_left, processed_right, sr=SAMPLE_RATE):
    """
    Compute PESQ for binaural signals (average of left and right channels)
    
    Args:
        clean_left, clean_right: Clean reference signals
        processed_left, processed_right: Processed signals (noisy or enhanced)
        sr: Sample rate
        
    Returns:
        PESQ score or None if calculation fails
    """
    try:
        # Try to import pesq
        from pesq import pesq
        
        # PESQ requires 16kHz or 8kHz
        if sr != 16000 and sr != 8000:
            # Resample to 16kHz
            clean_left = librosa.resample(clean_left, orig_sr=sr, target_sr=16000)
            clean_right = librosa.resample(clean_right, orig_sr=sr, target_sr=16000)
            processed_left = librosa.resample(processed_left, orig_sr=sr, target_sr=16000)
            processed_right = librosa.resample(processed_right, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Calculate PESQ for left and right channels
        try:
            pesq_left = pesq(sr, clean_left, processed_left, 'wb')
        except Exception as e:
            print(f"Error calculating PESQ for left channel: {e}")
            pesq_left = float('nan')
            
        try:
            pesq_right = pesq(sr, clean_right, processed_right, 'wb')
        except Exception as e:
            print(f"Error calculating PESQ for right channel: {e}")
            pesq_right = float('nan')
        
        # Return the average PESQ if at least one channel succeeded
        if not np.isnan(pesq_left) or not np.isnan(pesq_right):
            valid_scores = [score for score in [pesq_left, pesq_right] if not np.isnan(score)]
            return np.mean(valid_scores)
        else:
            return None
            
    except ImportError:
        print("PESQ calculation skipped: pesq package not installed. Install with 'pip install pesq'.")
        return None
    except Exception as e:
        print(f"Error in PESQ calculation: {e}")
        return None


# 8. Safe evaluation function that handles errors properly
def safe_evaluate_file(clean_path, noisy_path, model, device):
    """
    Safely evaluate a single file pair with proper error handling
    
    Args:
        clean_path: Path to clean reference file
        noisy_path: Path to noisy file
        model: Pytorch model
        device: Device to run model on
        
    Returns:
        Dictionary of evaluation metrics or None if evaluation fails
    """
    try:
        # Load files
        clean_data, c_sr = sf.read(clean_path)
        noisy_data, n_sr = sf.read(noisy_path)
        
        # Skip mono files
        if len(clean_data.shape) == 1 or len(noisy_data.shape) == 1:
            print(f"Skipping {clean_path.name}: mono file detected")
            return None
            
        # Skip if too short
        if len(clean_data) / c_sr < 0.3:  # Less than 300ms
            print(f"Skipping {clean_path.name}: too short ({len(clean_data) / c_sr:.2f}s)")
            return None
            
        # Check for silence
        clean_rms_l = np.sqrt(np.mean(clean_data[:, 0]**2))
        clean_rms_r = np.sqrt(np.mean(clean_data[:, 1]**2))
        if clean_rms_l < 0.001 or clean_rms_r < 0.001:  # -60 dBFS threshold
            print(f"Skipping {clean_path.name}: too quiet (RMS_L={20*np.log10(clean_rms_l+1e-10):.1f} dBFS, RMS_R={20*np.log10(clean_rms_r+1e-10):.1f} dBFS)")
            return None
            
        # Resample if needed
        if n_sr != c_sr:
            # Resample each channel separately
            noisy_data = np.stack([
                librosa.resample(noisy_data[:, ch], orig_sr=n_sr, target_sr=c_sr) 
                for ch in range(noisy_data.shape[1])
            ], axis=1)
            
        # Process through model - create batch tensor [1, 2, samples]
        noisy_tensor = torch.tensor(noisy_data.T, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Run through model
        with torch.no_grad():
            enhanced_tensor = model(noisy_tensor)
            enhanced_data = enhanced_tensor.cpu().numpy()[0]  # [2, samples]
            
        # Ensure enhanced data matches the original clean length
        orig_len = clean_data.shape[0]
        if enhanced_data.shape[1] > orig_len:
            enhanced_data = enhanced_data[:, :orig_len]
        elif enhanced_data.shape[1] < orig_len:
            pad_width = ((0, 0), (0, orig_len - enhanced_data.shape[1]))
            enhanced_data = np.pad(enhanced_data, pad_width)
            
        # Calculate metrics with proper error handling
        results = {
            'filename': clean_path.name
        }
        
        # MBSTOI for both noisy and enhanced signals
        noisy_mbstoi = safe_mbstoi(
            clean_data[:, 0], clean_data[:, 1], 
            noisy_data[:, 0], noisy_data[:, 1], 
            sr=c_sr
        )
        enhanced_mbstoi = safe_mbstoi(
            clean_data[:, 0], clean_data[:, 1], 
            enhanced_data[0], enhanced_data[1], 
            sr=c_sr
        )
        
        if noisy_mbstoi is not None:
            results['MBSTOI_noisy'] = noisy_mbstoi
        if enhanced_mbstoi is not None:
            results['MBSTOI'] = enhanced_mbstoi
        
        # PESQ for both noisy and enhanced signals
        try:
            noisy_pesq = safe_pesq(
                clean_data[:, 0], clean_data[:, 1],
                noisy_data[:, 0], noisy_data[:, 1],
                sr=c_sr
            )
            enhanced_pesq = safe_pesq(
                clean_data[:, 0], clean_data[:, 1],
                enhanced_data[0], enhanced_data[1],
                sr=c_sr
            )
            
            if noisy_pesq is not None:
                results['PESQ_noisy'] = noisy_pesq
            if enhanced_pesq is not None:
                results['PESQ'] = enhanced_pesq
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
            
        # Segmental SNR for both noisy and enhanced signals
        try:
            # Calculate absolute SegSNR for noisy signal
            noisy_segsnr_l = get_absolute_segsnr(clean_data[:, 0], noisy_data[:, 0], sr=c_sr)
            noisy_segsnr_r = get_absolute_segsnr(clean_data[:, 1], noisy_data[:, 1], sr=c_sr)
            results['SegSNR_noisy_L'] = noisy_segsnr_l
            results['SegSNR_noisy_R'] = noisy_segsnr_r
            results['SegSNR_noisy_Avg'] = (noisy_segsnr_l + noisy_segsnr_r) / 2
            
            # Calculate absolute SegSNR for enhanced signal
            enhanced_segsnr_l = get_absolute_segsnr(clean_data[:, 0], enhanced_data[0], sr=c_sr)
            enhanced_segsnr_r = get_absolute_segsnr(clean_data[:, 1], enhanced_data[1], sr=c_sr)
            results['SegSNR_L'] = enhanced_segsnr_l
            results['SegSNR_R'] = enhanced_segsnr_r
            results['SegSNR_Avg'] = (enhanced_segsnr_l + enhanced_segsnr_r) / 2
            
            # Calculate SegSNR improvement (for compatibility with paper)
            segsnr_l = fwsegsnr(clean_data[:, 0], noisy_data[:, 0], enhanced_data[0], sr=c_sr)
            segsnr_r = fwsegsnr(clean_data[:, 1], noisy_data[:, 1], enhanced_data[1], sr=c_sr)
            results['SegSNR_impr_L'] = segsnr_l
            results['SegSNR_impr_R'] = segsnr_r
            results['SegSNR_impr_Avg'] = (segsnr_l + segsnr_r) / 2
        except Exception as e:
            print(f"SegSNR calculation failed: {e}")
            
        # ILD error with mask for both noisy and enhanced
        try:
            noisy_ild_error = compute_ild_error_with_mask(
                clean_data[:, 0], clean_data[:, 1],
                noisy_data[:, 0], noisy_data[:, 1],
                sr=c_sr
            )
            enhanced_ild_error = compute_ild_error_with_mask(
                clean_data[:, 0], clean_data[:, 1],
                enhanced_data[0], enhanced_data[1],
                sr=c_sr
            )
            results['ILD_error_noisy'] = noisy_ild_error
            results['ILD_error'] = enhanced_ild_error
        except Exception as e:
            print(f"ILD error calculation failed: {e}")
            
        # IPD error with mask for both noisy and enhanced
        try:
            noisy_ipd_error = compute_ipd_error_with_mask(
                clean_data[:, 0], clean_data[:, 1],
                noisy_data[:, 0], noisy_data[:, 1],
                sr=c_sr
            )
            enhanced_ipd_error = compute_ipd_error_with_mask(
                clean_data[:, 0], clean_data[:, 1],
                enhanced_data[0], enhanced_data[1],
                sr=c_sr
            )
            results['IPD_error_noisy'] = noisy_ipd_error
            results['IPD_error'] = enhanced_ipd_error
            
            # Also convert IPD to degrees for easier interpretation
            results['IPD_error_noisy_deg'] = noisy_ipd_error * 180 / np.pi
            results['IPD_error_deg'] = enhanced_ipd_error * 180 / np.pi
        except Exception as e:
            print(f"IPD error calculation failed: {e}")
            
        return results
        
    except Exception as e:
        print(f"Failed to process {clean_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# 9. Main evaluation function that uses the improved metrics
def evaluate_model_improved(model_checkpoint, test_dataset_path, clean_dataset_path, 
                           output_dir, specific_snr=None, batch_size=8, 
                           limit_pairs=None, save_audio=True):
    """
    Evaluate model with improved metrics calculation to match the paper
    
    Args:
        model_checkpoint: Path to model checkpoint
        test_dataset_path: Path to noisy test dataset
        clean_dataset_path: Path to clean dataset
        output_dir: Directory to save results
        specific_snr: Filter by specific SNR value
        batch_size: Batch size for model inference
        limit_pairs: Limit number of pairs to evaluate (e.g. 750 as in paper)
        save_audio: Whether to save enhanced audio files
        
    Returns:
        Dictionary of evaluation results
    """
    # Initialize model
    import torch
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    
    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize(config_path="config")
    config = compose(config_name="config")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating on {device}")
    
    # Import model class - modify import based on your project structure
    from DCNN.trainer import DCNNLightningModule
    model = DCNNLightningModule(config)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if save_audio:
        os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
    
    # Find matching file pairs
    clean_files = list(Path(clean_dataset_path).glob('**/*.wav'))
    noisy_files = list(Path(test_dataset_path).glob('**/*.wav'))
    
    print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")

    # Parse file tags to ensure proper azimuth matching
    def parse_tag(stem):
        m = re.search(r'_az(-?\d+)(?:_snr([-+]?\d+(?:\.\d+)?))?$', stem)
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
    
    # Sort pairs for reproducibility
    pairs.sort(key=lambda x: str(x[0]))
    
    # Limit pairs to the specified number if requested
    if limit_pairs is not None and limit_pairs > 0 and limit_pairs < len(pairs):
        print(f"Limiting evaluation to {limit_pairs} pairs (out of {len(pairs)} available)")
        pairs = pairs[:limit_pairs]  # Take the first N pairs
        
    print(f"Found {len(pairs)} matching clean/noisy pairs (azimuth-aligned)")

    if len(pairs) == 0:
        print("ERROR: No matching clean/noisy pairs found! Check file naming patterns.")
        return None
    
    # Initialize results - add noisy metrics
    results = {
        'filename': [],
        'SegSNR_noisy_L': [],
        'SegSNR_noisy_R': [],
        'SegSNR_noisy_Avg': [],
        'SegSNR_L': [],
        'SegSNR_R': [],
        'SegSNR_Avg': [],
        'SegSNR_impr_L': [],
        'SegSNR_impr_R': [],
        'SegSNR_impr_Avg': [],
        'MBSTOI_noisy': [],
        'MBSTOI': [],
        'PESQ_noisy': [],
        'PESQ': [],
        'ILD_error_noisy': [],
        'ILD_error': [],
        'IPD_error_noisy': [],
        'IPD_error': [],
        'IPD_error_noisy_deg': [],
        'IPD_error_deg': []
    }
    
    # Process in batches
    total_pairs = len(pairs)
    processed_pairs = 0
    pbar = tqdm(total=total_pairs, desc="Processing files")
    
    for batch_idx in range(0, total_pairs, batch_size):
        batch_end = min(batch_idx + batch_size, total_pairs)
        batch_pairs = pairs[batch_idx:batch_end]
        
        # Process each file
        for clean_path, noisy_path in batch_pairs:
            pair_results = safe_evaluate_file(clean_path, noisy_path, model, device)
            
            if pair_results:
                # Add to results
                for key in results.keys():
                    if key in pair_results:
                        results[key].append(pair_results[key])
                    elif key != 'filename':
                        results[key].append(float('nan'))  # Add NaN for missing metrics
                
                # Save enhanced audio if requested
                if save_audio and 'enhanced_data' in pair_results:
                    enhanced_path = os.path.join(output_dir, "enhanced", f"enhanced_{clean_path.name}")
                    sf.write(enhanced_path, pair_results['enhanced_data'], pair_results.get('sr', SAMPLE_RATE))
            
            processed_pairs += 1
            pbar.update(1)
    
    pbar.close()
    
    # Check if we processed any files
    if len(results['filename']) == 0:
        print("Warning: No files were successfully processed!")
        return None
    
    # Create DataFrame for analysis
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics (excluding NaN values)
    average_results = {
        # Noisy signal metrics
        'Average SegSNR Noisy (L)': np.nanmean(results_df['SegSNR_noisy_L']),
        'Average SegSNR Noisy (R)': np.nanmean(results_df['SegSNR_noisy_R']),
        'Average SegSNR Noisy': np.nanmean(results_df['SegSNR_noisy_Avg']),
        'Average MBSTOI Noisy': np.nanmean(results_df['MBSTOI_noisy']),
        'Average PESQ Noisy': np.nanmean(results_df['PESQ_noisy']),
        'Average ILD Error Noisy (dB)': np.nanmean(results_df['ILD_error_noisy']),
        'Average IPD Error Noisy (rad)': np.nanmean(results_df['IPD_error_noisy']),
        'Average IPD Error Noisy (deg)': np.nanmean(results_df['IPD_error_noisy_deg']),
        
        # Enhanced signal metrics
        'Average SegSNR (L)': np.nanmean(results_df['SegSNR_L']),
        'Average SegSNR (R)': np.nanmean(results_df['SegSNR_R']),
        'Average SegSNR': np.nanmean(results_df['SegSNR_Avg']),
        'Average SegSNR Improvement': np.nanmean(results_df['SegSNR_impr_Avg']),
        'Average MBSTOI': np.nanmean(results_df['MBSTOI']),
        'Average PESQ': np.nanmean(results_df['PESQ']),
        'Average ILD Error (dB)': np.nanmean(results_df['ILD_error']),
        'Average IPD Error (rad)': np.nanmean(results_df['IPD_error']),
        'Average IPD Error (deg)': np.nanmean(results_df['IPD_error_deg']),
        
        # Processing stats
        'Files Processed': len(results_df),
        'Total Files': total_pairs,
        'Processing Rate': f"{len(results_df) / total_pairs * 100:.1f}%"
    }
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    with open(os.path.join(output_dir, "average_results.txt"), 'w') as f:
        # First write noisy metrics
        f.write("===== NOISY SIGNAL METRICS =====\n")
        for metric in [key for key in average_results.keys() if 'Noisy' in key]:
            value = average_results[metric]
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
                print(f"{metric}: {value:.4f}")
            else:
                f.write(f"{metric}: {value}\n")
                print(f"{metric}: {value}")
        
        # Then write enhanced metrics
        f.write("\n===== ENHANCED SIGNAL METRICS =====\n")
        for metric in [key for key in average_results.keys() if 'Noisy' not in key and key not in ['Files Processed', 'Total Files', 'Processing Rate']]:
            value = average_results[metric]
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
                print(f"{metric}: {value:.4f}")
            else:
                f.write(f"{metric}: {value}\n")
                print(f"{metric}: {value}")
        
        # Finally write processing stats
        f.write("\n===== PROCESSING STATS =====\n")
        for metric in ['Files Processed', 'Total Files', 'Processing Rate']:
            value = average_results[metric]
            f.write(f"{metric}: {value}\n")
            print(f"{metric}: {value}")
            
    return average_results


# 10. Paper-style evaluation function with fixed SNR levels
def evaluate_at_paper_snrs(model_checkpoint, vctk_test_dir, vctk_clean_dir, 
                          timit_test_dir=None, timit_clean_dir=None, 
                          output_dir="results", batch_size=8, limit_pairs=750, 
                          save_audio=False):
    """
    Evaluate model at the same SNR levels used in the paper
    
    Args:
        model_checkpoint: Path to model checkpoint
        vctk_test_dir: Path to VCTK noisy test dataset
        vctk_clean_dir: Path to VCTK clean dataset
        timit_test_dir: Optional path to TIMIT noisy test dataset
        timit_clean_dir: Optional path to TIMIT clean dataset
        output_dir: Directory to save results
        batch_size: Batch size for model inference
        limit_pairs: Limit number of pairs to evaluate (750 as in paper)
        save_audio: Whether to save enhanced audio files
        
    Returns:
        Dictionary of results by SNR
    """
    # SNR levels used in the paper
    paper_snrs = [-6, -3, 0, 3, 6, 9, 12, 15]
    
    # Process VCTK dataset
    vctk_results_by_snr = {}
    vctk_output_dir = os.path.join(output_dir, "vctk")
    
    # Create output directory
    os.makedirs(vctk_output_dir, exist_ok=True)
    
    print("\n===== EVALUATING VCTK DATASET =====")
    for snr in paper_snrs:
        print(f"\n===== Evaluating VCTK at SNR = {snr} dB =====")
        snr_output_dir = os.path.join(vctk_output_dir, f"snr_{snr}dB")
        
        # Check if SNR-specific subdirectory exists
        snr_test_dir = os.path.join(vctk_test_dir, f"snr_{snr}dB")
        if not os.path.exists(snr_test_dir):
            print(f"SNR subdirectory not found: {snr_test_dir}")
            print(f"Using flat structure with specific_snr parameter...")
            snr_test_dir = vctk_test_dir
            
        # Evaluate at this SNR level
        snr_results = evaluate_model_improved(
            model_checkpoint,
            snr_test_dir,
            vctk_clean_dir,
            snr_output_dir,
            specific_snr=snr,
            batch_size=batch_size,
            limit_pairs=limit_pairs,
            save_audio=save_audio
        )
        
        if snr_results:
            vctk_results_by_snr[snr] = snr_results
    
    # Process TIMIT dataset if provided
    timit_results_by_snr = {}
    if timit_test_dir and timit_clean_dir:
        timit_output_dir = os.path.join(output_dir, "timit")
        
        # Create output directory
        os.makedirs(timit_output_dir, exist_ok=True)
        
        print("\n===== EVALUATING TIMIT DATASET (UNMATCHED CONDITION) =====")
        for snr in paper_snrs:
            print(f"\n===== Evaluating TIMIT at SNR = {snr} dB =====")
            snr_output_dir = os.path.join(timit_output_dir, f"snr_{snr}dB")
            
            # Check if SNR-specific subdirectory exists
            snr_test_dir = os.path.join(timit_test_dir, f"snr_{snr}dB")
            if not os.path.exists(snr_test_dir):
                print(f"SNR subdirectory not found: {snr_test_dir}")
                print(f"Using flat structure with specific_snr parameter...")
                snr_test_dir = timit_test_dir
                
            # Evaluate at this SNR level
            snr_results = evaluate_model_improved(
                model_checkpoint,
                snr_test_dir,
                timit_clean_dir,
                snr_output_dir,
                specific_snr=snr,
                batch_size=batch_size,
                limit_pairs=limit_pairs,
                save_audio=save_audio
            )
            
            if snr_results:
                timit_results_by_snr[snr] = snr_results
    
    # Generate paper-style tables for both datasets
    generate_paper_style_table(vctk_results_by_snr, os.path.join(output_dir, "vctk_paper_style_results.csv"), "VCTK")
    if timit_test_dir and timit_clean_dir:
        generate_paper_style_table(timit_results_by_snr, os.path.join(output_dir, "timit_paper_style_results.csv"), "TIMIT")
    
    # Combine results for overall average
    all_results = {
        'vctk': vctk_results_by_snr,
        'timit': timit_results_by_snr if timit_test_dir and timit_clean_dir else None
    }
    
    return all_results


def generate_paper_style_table(results_by_snr, output_file, dataset_name):
    """Generate paper-style results table for a dataset"""
    
    paper_snrs = [-6, -3, 0, 3, 6, 9, 12, 15]
    
    # Generate paper-style table for noisy and enhanced metrics
    with open(output_file, 'w') as f:
        # Header
        f.write("SNR,MBSTOI_noisy,MBSTOI,SegSNR_noisy,SegSNR,SegSNR_impr,L_ILD_noisy,L_ILD,L_IPD_noisy,L_IPD\n")
        
        for snr in paper_snrs:
            if snr in results_by_snr:
                result = results_by_snr[snr]
                f.write(f"{snr}," +
                        f"{result.get('Average MBSTOI Noisy', 'N/A'):.4f}," +
                        f"{result.get('Average MBSTOI', 'N/A'):.4f}," +
                        f"{result.get('Average SegSNR Noisy', 'N/A'):.2f}," +
                        f"{result.get('Average SegSNR', 'N/A'):.2f}," +
                        f"{result.get('Average SegSNR Improvement', 'N/A'):.2f}," +
                        f"{result.get('Average ILD Error Noisy (dB)', 'N/A'):.4f}," +
                        f"{result.get('Average ILD Error (dB)', 'N/A'):.4f}," +
                        f"{result.get('Average IPD Error Noisy (deg)', 'N/A'):.2f}," +
                        f"{result.get('Average IPD Error (deg)', 'N/A'):.2f}\n")
    
    # Calculate overall averages
    if results_by_snr:
        avg_mbstoi_noisy = np.nanmean([r.get('Average MBSTOI Noisy', 0) for r in results_by_snr.values()])
        avg_mbstoi = np.nanmean([r.get('Average MBSTOI', 0) for r in results_by_snr.values()])
        avg_segsnr_noisy = np.nanmean([r.get('Average SegSNR Noisy', 0) for r in results_by_snr.values()])
        avg_segsnr = np.nanmean([r.get('Average SegSNR', 0) for r in results_by_snr.values()])
        avg_segsnr_impr = np.nanmean([r.get('Average SegSNR Improvement', 0) for r in results_by_snr.values()])
        avg_ild_noisy = np.nanmean([r.get('Average ILD Error Noisy (dB)', 0) for r in results_by_snr.values()])
        avg_ild = np.nanmean([r.get('Average ILD Error (dB)', 0) for r in results_by_snr.values()])
        avg_ipd_noisy_deg = np.nanmean([r.get('Average IPD Error Noisy (deg)', 0) for r in results_by_snr.values()])
        avg_ipd_deg = np.nanmean([r.get('Average IPD Error (deg)', 0) for r in results_by_snr.values()])
        
        print(f"\n===== {dataset_name} PAPER STYLE AVERAGES =====")
        print(f"Average MBSTOI Noisy: {avg_mbstoi_noisy:.4f}")
        print(f"Average MBSTOI Enhanced: {avg_mbstoi:.4f}")
        print(f"Average SegSNR Noisy: {avg_segsnr_noisy:.2f}")
        print(f"Average SegSNR Enhanced: {avg_segsnr:.2f}")
        print(f"Average SegSNR Improvement: {avg_segsnr_impr:.2f}")
        print(f"Average ILD Error Noisy: {avg_ild_noisy:.4f} dB")
        print(f"Average ILD Error Enhanced: {avg_ild:.4f} dB")
        print(f"Average IPD Error Noisy: {avg_ipd_noisy_deg:.2f} deg")
        print(f"Average IPD Error Enhanced: {avg_ipd_deg:.2f} deg")
        
        # Save averages
        avg_file = output_file.replace('.csv', '_averages.txt')
        with open(avg_file, 'w') as f:
            f.write(f"===== {dataset_name} PAPER STYLE AVERAGES =====\n")
            f.write(f"Average MBSTOI Noisy: {avg_mbstoi_noisy:.4f}\n")
            f.write(f"Average MBSTOI Enhanced: {avg_mbstoi:.4f}\n")
            f.write(f"Average SegSNR Noisy: {avg_segsnr_noisy:.2f}\n")
            f.write(f"Average SegSNR Enhanced: {avg_segsnr:.2f}\n")
            f.write(f"Average SegSNR Improvement: {avg_segsnr_impr:.2f}\n")
            f.write(f"Average ILD Error Noisy: {avg_ild_noisy:.4f} dB\n")
            f.write(f"Average ILD Error Enhanced: {avg_ild:.4f} dB\n")
            f.write(f"Average IPD Error Noisy: {avg_ipd_noisy_deg:.2f} deg\n")
            f.write(f"Average IPD Error Enhanced: {avg_ipd_deg:.2f} deg\n")


# 11. Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate binaural speech enhancement model")
    
    # Basic evaluation arguments
    parser.add_argument("--model_checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dataset_path", help="Path to test dataset directory")
    parser.add_argument("--clean_dataset_path", help="Path to clean dataset directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    
    # Arguments for paper-style evaluation
    parser.add_argument("--paper_style_eval", action="store_true", help="Run paper-style evaluation with SNR breakdown")
    parser.add_argument("--vctk_test_dir", help="Path to VCTK noisy test dataset")
    parser.add_argument("--vctk_clean_dir", help="Path to VCTK clean dataset")
    parser.add_argument("--timit_test_dir", help="Path to TIMIT noisy test dataset")
    parser.add_argument("--timit_clean_dir", help="Path to TIMIT clean dataset")
    parser.add_argument("--specific_snr", type=float, help="Filter by specific SNR value")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for model inference (higher values use more GPU memory)")
    parser.add_argument("--limit_pairs", type=int, default=750, 
                        help="Limit number of pairs to evaluate (default: 750 as in paper)")
    parser.add_argument("--save_audio", action="store_true", help="Save enhanced audio files")
    
    args = parser.parse_args()
    
    if args.paper_style_eval:
        # Run paper-style evaluation with SNR breakdown (-6 to 15 dB)
        if not all([args.vctk_test_dir, args.vctk_clean_dir]):
            parser.error("Paper-style evaluation requires at least VCTK test and clean dataset paths")
        
        print("Running paper-style evaluation with SNR points from -6 to 15 dB")
        evaluate_at_paper_snrs(
            args.model_checkpoint,
            args.vctk_test_dir,
            args.vctk_clean_dir,
            args.timit_test_dir,
            args.timit_clean_dir,
            args.output_dir,
            batch_size=args.batch_size,
            limit_pairs=args.limit_pairs,
            save_audio=args.save_audio
        )
    else:
        # Run basic evaluation
        if not all([args.test_dataset_path, args.clean_dataset_path]):
            parser.error("Basic evaluation requires test and clean dataset paths")
        
        evaluate_model_improved(
            args.model_checkpoint,
            args.test_dataset_path,
            args.clean_dataset_path,
            args.output_dir,
            specific_snr=args.specific_snr,
            batch_size=args.batch_size,
            limit_pairs=args.limit_pairs,
            save_audio=args.save_audio
        )