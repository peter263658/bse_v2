def verify_dataset(clean_dir, noisy_dir, num_samples=3):
    """Check HRIR application and file pairing in the dataset"""
    import soundfile as sf
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from pathlib import Path
    
    # Get all clean and noisy files
    # clean_files = list(Path(clean_dir).glob('**/*.wav'))
    # noisy_files = list(Path(noisy_dir).glob('**/*.wav'))
    clean_files = list(Path(clean_dir).rglob('*.wav'))
    noisy_files = list(Path(noisy_dir).rglob('*.wav'))
    
    print(f"Found {len(clean_files)} clean and {len(noisy_files)} noisy files")
    
    # Create a proper matching
    # clean_dict = {f.stem.split('_az')[0]: f for f in clean_files}
    # noisy_dict = {}
    import re

    def get_base(stem: str):
        return re.sub(r'_az-?\d+(?:_snr[+\-]?\d+(?:\.\d+)?)?$','', stem)

    clean_dict = { get_base(f.stem): f for f in clean_files }
    noisy_dict = {}


    for f in noisy_files:
        # base_name = f.stem.split('_az')[0]
        base_name = get_base(f.stem)
        if base_name in noisy_dict:
            print(f"Warning: Duplicate base name {base_name} in noisy files")
        noisy_dict[base_name] = f
    
    # Find common base names
    common_keys = set(clean_dict.keys()) & set(noisy_dict.keys())
    print(f"Found {len(common_keys)} matching clean/noisy pairs")
    
    if len(common_keys) == 0:
        print("ERROR: No matching clean/noisy files found!")
        return
    
    # Check a few random files
    sample_keys = random.sample(list(common_keys), min(num_samples, len(common_keys)))
    
    for key in sample_keys:
        clean_path = clean_dict[key]
        noisy_path = noisy_dict[key]
        
        # Read files
        clean_data, c_sr = sf.read(clean_path)
        noisy_data, n_sr = sf.read(noisy_path)
        
        print(f"\nExamining pair: {clean_path.name} and {noisy_path.name}")
        print(f"- Sample rates: Clean {c_sr}Hz, Noisy {n_sr}Hz")
        
        # Check shapes
        print(f"- Clean shape: {clean_data.shape}")
        print(f"- Noisy shape: {noisy_data.shape}")
        
        # Check if truly binaural
        if len(clean_data.shape) == 2 and clean_data.shape[1] == 2:
            clean_corr = np.corrcoef(clean_data[:, 0], clean_data[:, 1])[0, 1]
            print(f"- Clean L/R correlation: {clean_corr:.4f}")
            
            # Check for extreme correlation (possible mono duplication)
            if abs(clean_corr) > 0.95:
                print("  WARNING: Clean channels have high correlation!")
                
            # Compute average ILD/IPD for clean file
            # Use a simple implementation for diagnostic purposes
            clean_stft_l = np.fft.rfft(clean_data[:, 0])
            clean_stft_r = np.fft.rfft(clean_data[:, 1])
            
            # ILD
            clean_ild = 20 * np.log10(np.abs(clean_stft_l) / (np.abs(clean_stft_r) + 1e-10))
            avg_ild = np.mean(np.abs(clean_ild))
            print(f"- Clean avg ILD: {avg_ild:.2f} dB")
            
            # IPD
            clean_ipd = np.angle(clean_stft_l * np.conj(clean_stft_r)) * 180 / np.pi
            avg_ipd = np.mean(np.abs(clean_ipd))
            print(f"- Clean avg IPD: {avg_ipd:.2f} degrees")
            
            # Values to look for:
            # - Good binaural: ILD ~1-5 dB, IPD ~10-60 degrees
            # - Mono duplication: ILD ~0 dB, IPD ~0 degrees
            # - Channel swapped: IPD values very high, possibly near 180
        
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
        
        # Plot clean spectrogram (left channel)
        plt.subplot(2, 2, 3)
        if len(clean_data.shape) == 2:
            plt.specgram(clean_data[:, 0], NFFT=512, Fs=c_sr)
        else:
            plt.specgram(clean_data, NFFT=512, Fs=c_sr)
        plt.title("Clean Spectrogram (Left Ch.)")
        
        # Plot noisy spectrogram (left channel)
        plt.subplot(2, 2, 4)
        if len(noisy_data.shape) == 2:
            plt.specgram(noisy_data[:, 0], NFFT=512, Fs=n_sr)
        else:
            plt.specgram(noisy_data, NFFT=512, Fs=n_sr)
        plt.title("Noisy Spectrogram (Left Ch.)")
        
        plt.tight_layout()
        plt.savefig(f"dataset_diagnostics_{key}.png")
        plt.close()


vctk_clean_dir = "/raid/R12K41024/BCCTN/Dataset/clean_testset"
vctk_test_dir = "/raid/R12K41024/BCCTN/Dataset/noisy_testset"
timit_clean_dir = "/raid/R12K41024/BCCTN/Dataset/clean_testset_timit"
timit_test_dir = "/raid/R12K41024/BCCTN/Dataset/noisy_testset_timit"



verify_dataset(vctk_clean_dir, vctk_test_dir)
verify_dataset(timit_clean_dir, timit_test_dir)