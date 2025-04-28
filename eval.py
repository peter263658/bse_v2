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

# Import custom modules
from DCNN.datasets.test_dataset import BaseDataset
from DCNN.trainer import DCNNLightningModule


def compute_segmental_snr(clean, noisy, enhanced, sr=16000):
    # Frame the signals
    frame_length = int(256 * sr / 16000)
    hop_length = int(128 * sr / 16000)
    
    clean_frames = librosa.util.frame(clean, frame_length=frame_length, hop_length=hop_length)
    noisy_frames = librosa.util.frame(noisy, frame_length=frame_length, hop_length=hop_length)
    enhanced_frames = librosa.util.frame(enhanced, frame_length=frame_length, hop_length=hop_length)
    
    # Compute SNR for each frame
    eps = 1e-10
    
    # Compute noise in noisy signal
    noise_noisy_frames = noisy_frames - clean_frames
    signal_power = np.sum(clean_frames**2, axis=0)
    noise_noisy_power = np.sum(noise_noisy_frames**2, axis=0) + eps
    
    # Skip frames with zero signal power to avoid log(0)
    valid_frames = signal_power > eps
    if not np.any(valid_frames):
        return 0.0  # Return 0 if no valid frames
    
    snr_noisy_frames = np.zeros_like(signal_power)
    snr_enhanced_frames = np.zeros_like(signal_power)
    
    # Only calculate SNR for frames with signal power
    snr_noisy_frames[valid_frames] = 10 * np.log10(
        signal_power[valid_frames] / noise_noisy_power[valid_frames]
    )
    
    # Compute noise in enhanced signal
    noise_enhanced_frames = enhanced_frames - clean_frames
    noise_enhanced_power = np.sum(noise_enhanced_frames**2, axis=0) + eps
    
    snr_enhanced_frames[valid_frames] = 10 * np.log10(
        signal_power[valid_frames] / noise_enhanced_power[valid_frames]
    )
    
    # Apply frequency weighting (simple approach)
    weights = np.linspace(1, 2, len(snr_noisy_frames))
    snr_noisy_frames = snr_noisy_frames * weights
    snr_enhanced_frames = snr_enhanced_frames * weights
    
    # Clip SNR values to reasonable range
    snr_noisy_frames = np.clip(snr_noisy_frames, -10, 35)
    snr_enhanced_frames = np.clip(snr_enhanced_frames, -10, 35)
    
    # Calculate SNR improvement, using only valid frames
    if np.sum(valid_frames) > 0:
        snr_improvement = (
            np.sum(snr_enhanced_frames[valid_frames]) / np.sum(valid_frames) - 
            np.sum(snr_noisy_frames[valid_frames]) / np.sum(valid_frames)
        )
    else:
        snr_improvement = 0.0
    
    return snr_improvement

def compute_ild_error(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """
    Compute ILD error between clean and enhanced binaural signals with improved mask
    """
    # Adjust STFT parameters based on sampling rate
    sr_ratio = sr / 16000
    n_fft = int(512 * sr_ratio)
    hop_length = int(100 * sr_ratio)
    win_length = int(400 * sr_ratio)
    
    # Compute STFTs
    clean_left_stft = librosa.stft(clean_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    clean_right_stft = librosa.stft(clean_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    enhanced_left_stft = librosa.stft(enhanced_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    enhanced_right_stft = librosa.stft(enhanced_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Compute ILD in dB
    eps = 1e-10
    clean_ild = 20 * np.log10(np.abs(clean_left_stft) / (np.abs(clean_right_stft) + eps) + eps)
    enhanced_ild = 20 * np.log10(np.abs(enhanced_left_stft) / (np.abs(enhanced_right_stft) + eps) + eps)
    
    # Create a better speech activity mask based on both left and right clean signals
    threshold = 20  # dB below max (as specified in the paper)
    
    # Combine energy from both channels for more robust mask
    combined_energy_left = np.abs(clean_left_stft)**2
    combined_energy_right = np.abs(clean_right_stft)**2
    combined_energy = np.maximum(combined_energy_left, combined_energy_right)
    
    max_energy = np.max(combined_energy)
    mask = (combined_energy > max_energy * 10**(-threshold/10))
    
    # Compute mean absolute error in active regions
    ild_error = np.abs(clean_ild - enhanced_ild)
    ild_error_masked = ild_error * mask
    
    # Return mean over active regions
    return np.sum(ild_error_masked) / (np.sum(mask) + eps)

def compute_ipd_error(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """
    Compute IPD error between clean and enhanced binaural signals with improved masking
    
    Args:
        clean_left: Clean left channel signal
        clean_right: Clean right channel signal
        enhanced_left: Enhanced left channel signal
        enhanced_right: Enhanced right channel signal
        sr: Sampling rate
    """
    # Adjust STFT parameters based on sampling rate
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
    
    # Create a better speech activity mask based on both left and right channels
    threshold = 20  # dB below max (as specified in the paper)
    
    # Combine energy from both channels for more robust mask
    combined_energy_left = np.abs(clean_left_stft)**2
    combined_energy_right = np.abs(clean_right_stft)**2
    combined_energy = np.maximum(combined_energy_left, combined_energy_right)
    
    max_energy = np.max(combined_energy)
    mask = (combined_energy > max_energy * 10**(-threshold/10))
    
    # Apply the mask only to frequency bands where IPD is meaningful
    # IPD is less reliable at very low and very high frequencies
    # You can adjust these frequency ranges based on your data
    f_min = 200  # Hz
    f_max = 8000  # Hz
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = np.logical_and(freqs >= f_min, freqs <= f_max)
    freq_mask = freq_mask[:, np.newaxis]  # Reshape for broadcasting
    combined_mask = np.logical_and(mask, freq_mask)
    
    # Handle phase wrapping by taking the smallest angle difference
    ipd_error = np.abs(np.angle(np.exp(1j * (clean_ipd - enhanced_ipd))))
    ipd_error_masked = ipd_error * combined_mask
    
    # Convert to degrees
    ipd_error_degrees = ipd_error_masked * (180 / np.pi)
    
    # Return mean over active regions
    total_active_bins = np.sum(combined_mask) + eps
    mean_ipd_error = np.sum(ipd_error_degrees) / total_active_bins
    
    return mean_ipd_error

def compute_mbstoi(clean_left, clean_right, enhanced_left, enhanced_right, sr=16000):
    """
    Compute MBSTOI metric for binaural signals as specified in the paper
    Uses the standard MBSTOI implementation (not mbstoi_beta)
    
    Reference:
    A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, "Refinement
    and validation of the binaural short time objective intelligibility
    measure for spatially diverse conditions," Speech Communication,
    vol. 102, pp. 1-13, Sep. 2018.
    
    Args:
        clean_left: Clean left channel signal
        clean_right: Clean right channel signal
        enhanced_left: Enhanced left channel signal
        enhanced_right: Enhanced right channel signal
        sr: Sampling rate of the signals (must be 16 kHz for MBSTOI)
    """
    # MBSTOI expects 10 kHz sampling rate
    target_sr = 10000
    
    # Resample if not already at 10 kHz
    if sr != target_sr:
        from scipy.signal import resample
        
        # Calculate new length
        new_length = int(len(clean_left) * target_sr / sr)
        
        # Resample all signals to 10 kHz
        clean_left = resample(clean_left, new_length)
        clean_right = resample(clean_right, new_length)
        enhanced_left = resample(enhanced_left, new_length)
        enhanced_right = resample(enhanced_right, new_length)
    
    try:
        from MBSTOI.mbstoi import mbstoi
        
        # The paper uses a coarseness value of 1 (default)
        score = mbstoi(clean_left, clean_right, enhanced_left, enhanced_right, gridcoarseness=1)
        return score
    except ImportError:
        print("MBSTOI module not found. Using simplified STOI calculation.")
        # Fallback to simplified STOI calculation
        
        # Try to use the stoi function from torchaudio if available
        try:
            import torchaudio.functional as F
            import torch
            
            # Convert to tensors
            c_l = torch.tensor(clean_left, dtype=torch.float32)
            c_r = torch.tensor(clean_right, dtype=torch.float32)
            e_l = torch.tensor(enhanced_left, dtype=torch.float32)
            e_r = torch.tensor(enhanced_right, dtype=torch.float32)
            
            # Compute STOI for each ear
            stoi_l = F.stoi(e_l, c_l, target_sr, extended=False)
            stoi_r = F.stoi(e_r, c_r, target_sr, extended=False)
            
            # Return average - this is a simplified approximation, not true MBSTOI
            return (stoi_l.item() + stoi_r.item()) / 2
            
        except (ImportError, AttributeError):
            # Fallback to very basic approximation
            print("WARNING: Using very basic STOI approximation - results will not be comparable to the paper.")
            stoi_left = librosa.feature.spectral_flatness(y=clean_left - enhanced_left).mean()
            stoi_right = librosa.feature.spectral_flatness(y=clean_right - enhanced_right).mean()
            
            # Invert and normalize to [0, 1] range (higher is better)
            score = 1 - (stoi_left + stoi_right) / 2
            return score

def evaluate_model(model_checkpoint, test_dataset_path, clean_dataset_path, output_dir, 
                is_timit=False, specific_snr=None):
    """
    Evaluate trained model on test dataset
    
    Args:
        model_checkpoint: Path to model checkpoint
        test_dataset_path: Path to test dataset directory
        clean_dataset_path: Path to clean dataset directory
        output_dir: Directory to save results
        is_timit: Whether this is TIMIT (unmatched condition) evaluation
        specific_snr: Specific SNR to evaluate (for paper-style evaluation)
    """
    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize(config_path="config")
    config = compose(config_name="config")
    
    # Create test dataset
    # If evaluating a specific SNR, filter files by SNR value
    if specific_snr is not None:
        # Custom dataset that filters by SNR
        class SNRFilteredDataset(BaseDataset):
            def __init__(self, noisy_dir, clean_dir, target_snr):
                super().__init__(noisy_dir, clean_dir)
                self.target_snr = target_snr
                self._filter_by_snr()
                
            def _filter_by_snr(self):
                # Filter files based on SNR in filename
                filtered_noisy_paths = []
                filtered_clean_paths = []
                
                for i, noisy_path in enumerate(self.noisy_file_paths):
                    filename = os.path.basename(noisy_path)
                    # Check if SNR is in the filename (format: *_snr{value}.wav)
                    if f"_snr{self.target_snr}." in filename or f"_snr{self.target_snr:.1f}." in filename:
                        filtered_noisy_paths.append(noisy_path)
                        filtered_clean_paths.append(self.target_file_paths[i])
                
                self.noisy_file_paths = filtered_noisy_paths
                self.target_file_paths = filtered_clean_paths
                
                print(f"Filtered to {len(self.noisy_file_paths)} files with SNR {self.target_snr} dB")
        
        dataset = SNRFilteredDataset(test_dataset_path, clean_dataset_path, specific_snr)
    else:
        dataset = BaseDataset(test_dataset_path, clean_dataset_path)
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=2
    )
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating on {device }")
    model = DCNNLightningModule(config)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'filename': [],
        'segSNR_L': [],
        'segSNR_R': [],
        'MBSTOI': [],
        'ILD_error': [],
        'IPD_error': []
    }
    
    # Process each file
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        noisy, clean, clean_path, noisy_path = batch
        
        # Process with model
        with torch.no_grad():
            noisy = noisy.to(device)
            enhanced = model(noisy)
            
        # Check sampling rate from audio files to ensure accurate metric calculations
        if isinstance(clean_path, list) and len(clean_path) > 0:
            # Get the sampling rate from the first file
            _, file_sr = librosa.load(clean_path[0], sr=None, mono=False, duration=0.1)
        else:
            # Default to 16kHz if we can't determine
            file_sr = 16000
            
        # print(f"File sampling rate: {file_sr} Hz")
        
        # Convert to numpy for metrics calculation
        noisy_np = noisy.cpu().numpy()[0]  # [channels, samples]
        clean_np = clean.cpu().numpy()[0]  # [channels, samples]
        enhanced_np = enhanced.cpu().numpy()[0]  # [channels, samples]
        
        # Get filename
        filename = os.path.basename(clean_path[0])
        results['filename'].append(filename)
        
        # Calculate metrics
        # Segmental SNR
        segSNR_L = compute_segmental_snr(clean_np[0], noisy_np[0], enhanced_np[0], sr=file_sr)
        segSNR_R = compute_segmental_snr(clean_np[1], noisy_np[1], enhanced_np[1], sr=file_sr)
        results['segSNR_L'].append(segSNR_L)
        results['segSNR_R'].append(segSNR_R)
        
        # MBSTOI
        mbstoi_score = compute_mbstoi(clean_np[0], clean_np[1], enhanced_np[0], enhanced_np[1], sr=file_sr)
        results['MBSTOI'].append(mbstoi_score)
        
        # ILD error
        ild_error = compute_ild_error(clean_np[0], clean_np[1], enhanced_np[0], enhanced_np[1], sr=file_sr)
        results['ILD_error'].append(ild_error)
        
        # IPD error
        ipd_error = compute_ipd_error(clean_np[0], clean_np[1], enhanced_np[0], enhanced_np[1], sr=file_sr)
        results['IPD_error'].append(ipd_error)
        
        # Save enhanced audio
        enhanced_path = os.path.join(output_dir, "enhanced", f"enhanced_{filename}")
        sf.write(enhanced_path, enhanced_np.T, 16000)
        
        # Save a few spectrograms for visualization
        if i < 5:  # Only save first 5 examples
            plt.figure(figsize=(15, 10))
            
            # Clean spectrogram - Left channel
            plt.subplot(3, 2, 1)
            clean_spec_l = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np[0])), ref=np.max)
            librosa.display.specshow(clean_spec_l, y_axis='log', x_axis='time', sr=16000)
            plt.title('Clean - Left Channel')
            plt.colorbar(format='%+2.0f dB')
            
            # Clean spectrogram - Right channel
            plt.subplot(3, 2, 2)
            clean_spec_r = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np[1])), ref=np.max)
            librosa.display.specshow(clean_spec_r, y_axis='log', x_axis='time', sr=16000)
            plt.title('Clean - Right Channel')
            plt.colorbar(format='%+2.0f dB')
            
            # Noisy spectrogram - Left channel
            plt.subplot(3, 2, 3)
            noisy_spec_l = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np[0])), ref=np.max)
            librosa.display.specshow(noisy_spec_l, y_axis='log', x_axis='time', sr=16000)
            plt.title('Noisy - Left Channel')
            plt.colorbar(format='%+2.0f dB')
            
            # Noisy spectrogram - Right channel
            plt.subplot(3, 2, 4)
            noisy_spec_r = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np[1])), ref=np.max)
            librosa.display.specshow(noisy_spec_r, y_axis='log', x_axis='time', sr=16000)
            plt.title('Noisy - Right Channel')
            plt.colorbar(format='%+2.0f dB')
            
            # Enhanced spectrogram - Left channel
            plt.subplot(3, 2, 5)
            enhanced_spec_l = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_np[0])), ref=np.max)
            librosa.display.specshow(enhanced_spec_l, y_axis='log', x_axis='time', sr=16000)
            plt.title('Enhanced - Left Channel')
            plt.colorbar(format='%+2.0f dB')
            
            # Enhanced spectrogram - Right channel
            plt.subplot(3, 2, 6)
            enhanced_spec_r = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_np[1])), ref=np.max)
            librosa.display.specshow(enhanced_spec_r, y_axis='log', x_axis='time', sr=16000)
            plt.title('Enhanced - Right Channel')
            plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"spectrogram_{i}_{filename}.png"))
            plt.close()
    
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
    
    # Create plots
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['segSNR_L'], bins=20, alpha=0.5, label='Left')
    plt.hist(results_df['segSNR_R'], bins=20, alpha=0.5, label='Right')
    plt.xlabel('Segmental SNR (dB)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Segmental SNR Improvement')
    plt.savefig(os.path.join(output_dir, "segSNR_distribution.png"))
    
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['MBSTOI'], bins=20)
    plt.xlabel('MBSTOI Score')
    plt.ylabel('Count')
    plt.title('Distribution of MBSTOI Scores')
    plt.savefig(os.path.join(output_dir, "mbstoi_distribution.png"))
    
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['ILD_error'], bins=20)
    plt.xlabel('ILD Error (dB)')
    plt.ylabel('Count')
    plt.title('Distribution of ILD Errors')
    plt.savefig(os.path.join(output_dir, "ild_error_distribution.png"))
    
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['IPD_error'], bins=20)
    plt.xlabel('IPD Error (degrees)')
    plt.ylabel('Count')
    plt.title('Distribution of IPD Errors')
    plt.savefig(os.path.join(output_dir, "ipd_error_distribution.png"))
    
    return average_results

def run_paper_style_evaluation(model_checkpoint, vctk_test_dir, vctk_clean_dir, 
                          timit_test_dir, timit_clean_dir, output_dir):
    """
    Run evaluation in the same style as presented in the paper:
    - Test on VCTK (matched speakers)
    - Test on TIMIT (unmatched speakers)
    - Separate results by SNR values (-6, -3, 0, 3, 6, 9, 12, 15 dB)
    - Calculate metrics for each condition
    
    Args:
        model_checkpoint: Path to model checkpoint
        vctk_test_dir: Path to VCTK noisy test dataset 
        vctk_clean_dir: Path to VCTK clean test dataset
        timit_test_dir: Path to TIMIT noisy test dataset
        timit_clean_dir: Path to TIMIT clean test dataset
        output_dir: Directory to save results
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define SNR values as used in the paper
    snr_values = [-6, -3, 0, 3, 6, 9, 12, 15]
    
    # Prepare results dictionaries for paper tables
    vctk_results = {snr: {} for snr in snr_values}
    timit_results = {snr: {} for snr in snr_values}
    
    # VCTK matched condition testing
    print("=== Evaluating on VCTK (matched condition) ===")
    
    for snr in snr_values:
        print(f"\nEvaluating VCTK at SNR = {snr} dB")
        # Filter files by SNR
        snr_output_dir = os.path.join(output_dir, f"vctk_snr_{snr}dB")
        
        # Run evaluation at this SNR level
        vctk_results[snr] = evaluate_model(
            model_checkpoint,
            vctk_test_dir,
            vctk_clean_dir,
            snr_output_dir,
            is_timit=False,
            specific_snr=snr
        )
    
    # TIMIT unmatched condition testing (if available)
    if timit_test_dir and timit_clean_dir:
        print("\n=== Evaluating on TIMIT (unmatched condition) ===")
        
        for snr in snr_values:
            print(f"\nEvaluating TIMIT at SNR = {snr} dB")
            # SNR-specific directory for TIMIT
            timit_snr_test_dir = os.path.join(timit_test_dir, f"snr_{snr}dB")
            
            if not os.path.exists(timit_snr_test_dir):
                print(f"Warning: {timit_snr_test_dir} not found. Skipping TIMIT evaluation at {snr} dB.")
                continue
                
            snr_output_dir = os.path.join(output_dir, f"timit_snr_{snr}dB")
            
            # Run evaluation at this SNR level
            timit_results[snr] = evaluate_model(
                model_checkpoint,
                timit_snr_test_dir,
                timit_clean_dir,
                snr_output_dir,
                is_timit=True,
                specific_snr=snr
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