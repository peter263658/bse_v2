import soundfile as sf
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
import librosa
import re
from collections import defaultdict
import argparse

def verify_dataset(clean_dir, noisy_dir, num_samples=5, specific_snr=None):
    """Check HRIR application and file pairing in the dataset"""
    
    # Find all files - using rglob instead of glob to search subdirectories
    clean_files = list(Path(clean_dir).rglob('*.wav'))
    noisy_files = list(Path(noisy_dir).rglob('*.wav'))
    
    print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")
    
    # Use regex to match filenames with azimuth and SNR
    clean_pattern = re.compile(r'(.+?)_az(-?\d+)(?:_snr[-+]?\d+(?:\.\d+)?)?\.wav$', re.I)
    noisy_pattern = re.compile(r'(.+?)_az(-?\d+)_snr([-+]?\d+(?:\.\d+)?)\.wav$', re.I)
    
    # Create noisy file index {base_name: {azimuth: {snr: path}}}
    noisy_index = defaultdict(lambda: defaultdict(dict))
    for f in noisy_files:
        m = noisy_pattern.match(f.name)
        if m:
            base_name, azimuth, snr = m.groups()
            noisy_index[base_name][azimuth][float(snr)] = f
    
    # Find matching clean/noisy pairs
    pairs = []
    for cf in clean_files:
        m = clean_pattern.match(cf.name)
        if m:
            base_name, azimuth = m.groups()
            
            # If specific SNR is provided, only match that SNR
            if specific_snr is not None:
                noisy_path = noisy_index.get(base_name, {}).get(azimuth, {}).get(float(specific_snr))
                if noisy_path:
                    pairs.append((cf, noisy_path))
            else:
                # Otherwise match all SNRs
                for snr, nf in noisy_index.get(base_name, {}).get(azimuth, {}).items():
                    pairs.append((cf, nf))
    
    print(f"Found {len(pairs)} matching clean/noisy pairs")
    
    if len(pairs) == 0:
        print("ERROR: No matching clean/noisy pairs found! Check file naming patterns.")
        return
    
    # Check random samples
    sample_pairs = random.sample(pairs, min(num_samples, len(pairs)))
    
    for clean_path, noisy_path in sample_pairs:
        # Read files
        clean_data, c_sr = sf.read(clean_path)
        noisy_data, n_sr = sf.read(noisy_path)
        
        print(f"\nExamining pair: {clean_path.name} and {noisy_path.name}")
        print(f"- Sample rates: Clean {c_sr}Hz, Noisy {n_sr}Hz")
        
        # Check shapes
        print(f"- Clean shape: {clean_data.shape}")
        print(f"- Noisy shape: {noisy_data.shape}")
        
        # Check binaural channels
        if len(clean_data.shape) == 2 and clean_data.shape[1] == 2:
            # Calculate channel correlation
            clean_corr = np.corrcoef(clean_data[:, 0], clean_data[:, 1])[0, 1]
            print(f"- Clean L/R correlation: {clean_corr:.4f}")
            
            # Check for negative correlation (possible channel inversion)
            if clean_corr < -0.5:
                print("  WARNING: Clean channels have strong negative correlation! Possible phase inversion.")
            
            # Check for extreme correlation (possible mono duplication)
            if abs(clean_corr) > 0.95:
                print("  WARNING: Clean channels have extremely high correlation! Possible mono duplication.")
                
            # Calculate RMS of each channel
            clean_rms_l = np.sqrt(np.mean(clean_data[:, 0]**2))
            clean_rms_r = np.sqrt(np.mean(clean_data[:, 1]**2))
            
            # Compute overall ILD (in dB)
            clean_ild_db = 20 * np.log10(clean_rms_l / (clean_rms_r + 1e-10))
            print(f"- Clean overall ILD: {clean_ild_db:.2f} dB")
            
            # Compute STFT for more detailed analysis
            n_fft = 512
            hop_length = 128
            
            clean_stft_l = librosa.stft(clean_data[:, 0], n_fft=n_fft, hop_length=hop_length)
            clean_stft_r = librosa.stft(clean_data[:, 1], n_fft=n_fft, hop_length=hop_length)
            
            # Compute frequency-dependent ILD (in dB)
            clean_mag_l = np.abs(clean_stft_l) + 1e-10
            clean_mag_r = np.abs(clean_stft_r) + 1e-10
            
            freq_ild = 20 * np.log10(clean_mag_l / clean_mag_r)
            avg_freq_ild = np.mean(np.abs(freq_ild))
            print(f"- Clean frequency-dependent avg ILD: {avg_freq_ild:.2f} dB")
            
            # Compute IPD (in degrees) with energy thresholding
            energy_threshold = -40  # dB
            
            # Calculate joint energy for masking low-energy frames
            combined_energy = (clean_mag_l**2 + clean_mag_r**2) / 2
            max_energy = np.max(combined_energy)
            mask = combined_energy > (max_energy * 10**(energy_threshold/10))
            
            # Only calculate IPD in high-energy regions
            clean_ipd_rad = np.angle(clean_stft_l * np.conj(clean_stft_r))
            clean_ipd_deg = np.rad2deg(clean_ipd_rad)
            
            # Apply mask to get more accurate statistics
            masked_ipd = clean_ipd_deg[mask]
            
            # If not enough high-energy frames, use all frames
            if np.sum(mask) < (clean_ipd_deg.size * 0.05):  # At least 5% of frames
                print("  NOTE: Not enough high-energy frames found, using all frames for IPD calculation.")
                avg_ipd = np.mean(np.abs(clean_ipd_deg))
            else:
                avg_ipd = np.mean(np.abs(masked_ipd)) if masked_ipd.size > 0 else 0
                
            print(f"- Clean avg IPD: {avg_ipd:.2f} degrees")
            
            # Provide appropriate warning based on IPD values
            if avg_ipd > 90 and np.sum(mask) > (clean_ipd_deg.size * 0.05):
                print("  NOTE: High IPD values, but consistent with expected spatial cues for this azimuth.")
            elif avg_ipd > 90:
                print("  WARNING: Very large average IPD values! Possible phase issues.")
            
            # Calculate noise channel correlation
            if len(noisy_data.shape) == 2 and noisy_data.shape[1] == 2:
                noisy_corr = np.corrcoef(noisy_data[:, 0], noisy_data[:, 1])[0, 1]
                print(f"- Noisy L/R correlation: {noisy_corr:.4f}")
            
            # Check reasonable ILD values
            if abs(clean_ild_db) > 20:
                print("  WARNING: Extreme ILD value detected! One channel may be very quiet or missing.")
            
            # Display range statistics for active regions
            if np.sum(mask) > 0:
                masked_ild = freq_ild[mask]
                
                print(f"- Clean ILD range (active bins): {np.percentile(masked_ild, 5):.2f} to {np.percentile(masked_ild, 95):.2f} dB")
                print(f"- Clean IPD range (active bins): {np.percentile(masked_ipd, 5):.2f} to {np.percentile(masked_ipd, 95):.2f} degrees")
        
        # Generate diagnostic plot
        plt.figure(figsize=(12, 8))
        
        # Plot clean waveforms
        plt.subplot(2, 2, 1)
        if len(clean_data.shape) == 2:
            plt.plot(clean_data[:, 0], alpha=0.7, label='Left')
            plt.plot(clean_data[:, 1], alpha=0.7, label='Right')
        else:
            plt.plot(clean_data, label='Mono')
        plt.title("Clean Waveform")
        plt.legend()
        
        # Plot noisy waveforms
        plt.subplot(2, 2, 2)
        if len(noisy_data.shape) == 2:
            plt.plot(noisy_data[:, 0], alpha=0.7, label='Left')
            plt.plot(noisy_data[:, 1], alpha=0.7, label='Right')
        else:
            plt.plot(noisy_data, label='Mono')
        plt.title("Noisy Waveform")
        plt.legend()
        
        # Only plot ILD and IPD for binaural signals
        if len(clean_data.shape) == 2 and clean_data.shape[1] == 2:
            # Plot frequency-dependent ILD map
            plt.subplot(2, 2, 3)
            plt.imshow(freq_ild, aspect='auto', origin='lower', 
                       vmin=-20, vmax=20, cmap='coolwarm')
            plt.colorbar(label='ILD (dB)')
            plt.ylabel('Frequency bin')
            plt.xlabel('Time frame')
            plt.title("ILD (dB) Map")
            
            # Plot frequency-dependent IPD map (masked)
            plt.subplot(2, 2, 4)
            masked_ipd_plot = np.copy(clean_ipd_deg)
            masked_ipd_plot[~mask] = np.nan  # Only show high-energy regions
            plt.imshow(masked_ipd_plot, aspect='auto', origin='lower', 
                       vmin=-180, vmax=180, cmap='hsv')
            plt.colorbar(label='IPD (degrees)')
            plt.ylabel('Frequency bin')
            plt.xlabel('Time frame')
            plt.title("IPD (degrees) Map (high-energy only)")
        else:
            plt.subplot(2, 2, 3)
            plt.text(0.5, 0.5, "Mono signal - no ILD", ha='center', va='center')
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, "Mono signal - no IPD", ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f"dataset_diagnostics_{clean_path.stem}.png")
        plt.close()

    # Find unpaired files
    matched_clean = set(cp for cp, _ in pairs)
    matched_noisy = set(noisy_p for _, noisy_p in pairs)
    
    extra_clean = set(clean_files) - matched_clean
    extra_noisy   = set(noisy_files) - matched_noisy
    
    if extra_clean:
        print(f"\nFound {len(extra_clean)} clean files without matching noisy files")
        if len(extra_clean) < 10:
            for cp in extra_clean:
                print(f"- {cp.name}")
    
    if extra_noisy:
        print(f"\nFound {len(extra_noisy)} noisy files without matching clean files")
        if len(extra_noisy) < 10:
            for noisy_p in extra_noisy:    
                print(f"- {noisy_p.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset pairs and binaural properties")
    parser.add_argument("--clean_dir", required=True, help="Directory with clean audio files")
    parser.add_argument("--noisy_dir", required=True, help="Directory with noisy audio files")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to analyze")
    parser.add_argument("--snr", type=float, help="Filter by specific SNR value")
    
    args = parser.parse_args()
    
    verify_dataset(args.clean_dir, args.noisy_dir, args.samples, args.snr)