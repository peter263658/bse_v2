import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
from scipy.io import loadmat
import librosa
import random
from pathlib import Path
import tqdm
import argparse

def load_hrir(hrir_path, format='wav', target_sr=16000):
    """
    Load HRIRs from wav files or mat file and ensure proper sampling rate
    
    Args:
        hrir_path: Path to HRIR files
        format: 'wav' or 'mat'
        target_sr: Target sampling rate (16 kHz as per paper)
        
    Returns:
        hrirs_dict: Dictionary with azimuths as keys and HRIR arrays as values
    """
    hrirs_dict = {}
    print(f"Loading HRIRs from {hrir_path} with format {format}")
    
    if format == 'wav':
        # Load from directory of wav files
        for file_path in Path(hrir_path).glob('*.wav'):
            file_name = file_path.stem
            # Parse azimuth from filename (adjust based on your file naming convention)
            if 'az' in file_name:
                try:
                    # Extract azimuth from the filename
                    # Format: anechoic_distcm_<distance>_el_<elevation>_az_<azimuth>.wav
                    az_part = file_name.split('_az_')[-1]
                    az = int(az_part)
                    
                    # Load the audio file with all channels
                    audio, sr = librosa.load(file_path, sr=None, mono=False)
                    
                    # Check if resampling is needed
                    if sr != target_sr:
                        # print(f"Resampling HRIR from {sr} Hz to {target_sr} Hz")
                        # Resample if needed
                        resampled_audio = np.zeros((audio.shape[0], int(audio.shape[1] * target_sr / sr)))
                        for ch in range(audio.shape[0]):
                            resampled_audio[ch] = librosa.resample(audio[ch], orig_sr=sr, target_sr=target_sr)
                        audio = resampled_audio
                    
                    # According to docs, first two channels are left and right in-ear IRs
                    if audio.shape[0] >= 2:
                        # Take only the first two channels (left and right in-ear)
                        # Channel 0 is already left ear, channel 1 is already right ear
                        hrir = audio[:2]
                        hrirs_dict[az] = hrir
                        
                        # Log the first few
                        if len(hrirs_dict) <= 3:
                            print(f"Loaded HRIR for azimuth {az}°, shape: {hrir.shape}")
                            # Calculate and print average magnitude ratio (ILD)
                            l_energy = np.sum(hrir[0]**2)
                            r_energy = np.sum(hrir[1]**2)
                            ild_db = 10 * np.log10((l_energy + 1e-10) / (r_energy + 1e-10))
                            # Dynamic message based on azimuth
                            if az > 0:
                                print(f"  Average ILD: {ild_db:.2f} dB (expected < 0 for positive azimuth)")
                            elif az < 0:
                                print(f"  Average ILD: {ild_db:.2f} dB (expected > 0 for negative azimuth)")
                            else:
                                print(f"  Average ILD: {ild_db:.2f} dB (zero azimuth)")
                    else:
                        print(f"Warning: {file_name} doesn't have enough channels (found {audio.shape[0]}, expected at least 2)")
                        
                except Exception as e:
                    print(f"Error loading HRIR file {file_path}: {e}")
    
    elif format == 'mat':
        # Load from MATLAB file
        try:
            mat_data = loadmat(hrir_path)
            # Adjust based on your specific mat file structure
            if 'hrir_data' in mat_data:
                hrir_data = mat_data['hrir_data']
                azimuths = mat_data.get('azimuths', np.arange(-90, 91, 5)) # Default: -90 to 90 in steps of 5
                
                # Check if sampling rate info is available
                orig_sr = mat_data.get('fs', 48000)  # Default to 48kHz if not specified
                
                # Resample if needed
                if orig_sr != target_sr:
                    print(f"Resampling HRIR from {orig_sr} Hz to {target_sr} Hz")
                    resampled_data = np.zeros((int(hrir_data.shape[0] * target_sr / orig_sr), hrir_data.shape[1], hrir_data.shape[2]))
                    
                    for i in range(hrir_data.shape[1]):  # Channels
                        for j in range(hrir_data.shape[2]):  # Positions
                            resampled_data[:, i, j] = librosa.resample(
                                hrir_data[:, i, j], orig_sr=orig_sr, target_sr=target_sr)
                    
                    hrir_data = resampled_data
                
                for i, az in enumerate(azimuths):
                    hrirs_dict[int(az)] = hrir_data[:, :, i]  # Assuming format is [samples, channels, positions]
        except Exception as e:
            print(f"Error loading HRIR from mat file: {e}")
    
    # If no HRIRs were loaded, create simple ones
    if len(hrirs_dict) == 0:
        print("WARNING: No HRIRs loaded! Creating simple HRIR for testing purposes only.")
        print("WARNING: Using simplified HRIRs will significantly affect model training quality!")
        print("WARNING: Please ensure correct HRIR files are available before actual training.")
        # Create simple HRIR for testing
        simple_hrir_length = 128
        for az in range(-90, 91, 5):
            # Create simple binaural HRIR with delay and level difference
            delay_samples = int(az * 0.05)  # Simplified ITD
            level_diff = az * 0.1  # Simplified ILD
            
            hrir_l = np.zeros(simple_hrir_length)
            hrir_r = np.zeros(simple_hrir_length)
            
            # Set impulse
            center_idx = simple_hrir_length // 2
            hrir_l[center_idx] = 1.0
            hrir_r[center_idx + np.sign(delay_samples) * min(abs(delay_samples), 20)] = 1.0 * (1.0 - level_diff/20)
            
            hrirs_dict[az] = np.vstack((hrir_l, hrir_r))
    
    print(f"Loaded {len(hrirs_dict)} HRIR azimuth positions")

    # 1) Filter to frontal plane first
    hrirs_dict = {az: hrir for az, hrir in hrirs_dict.items() if -90 <= az <= 90}

    # 2) Then validate and do channel swap if needed
    valid_hrirs_dict = {}
    for az, hrir in hrirs_dict.items():
        # Check if HRIR has 2 channels
        if hrir.shape[0] != 2:
            print(f"Warning: HRIR for azimuth {az}° has {hrir.shape[0]} channels, expected 2. Skipping.")
            continue
            
        # Check if channels have sufficient energy
        l_energy = np.sum(hrir[0]**2)
        r_energy = np.sum(hrir[1]**2)
        
        if l_energy < 1e-8 or r_energy < 1e-8:
            print(f"Warning: HRIR for azimuth {az}° has low energy: L={l_energy}, R={r_energy}. Skipping.")
            continue
            
        # Verify ILD is reasonable for this azimuth
        ild_db = 10 * np.log10((l_energy + 1e-10) / (r_energy + 1e-10))
        
        # CORRECTED LOGIC:
        # Positive azimuth (to your right) → right ear louder → ILD negative
        # Negative azimuth (to your left) → left ear louder → ILD positive
        expected_sign = -1 if az > 0 else (1 if az < 0 else 0)
        actual_sign = 1 if ild_db > 0 else (-1 if ild_db < 0 else 0)
        
        # If azimuth is significant but ILD has wrong sign, swap channels
        if abs(az) > 10 and expected_sign * actual_sign < 0:
            print(f"Warning: HRIR for azimuth {az}° has unexpected ILD sign: {ild_db:.2f} dB. Swapping channels.")
            hrir = np.array([hrir[1], hrir[0]])  # Swap channels
            # Recalculate ILD
            l_energy = np.sum(hrir[0]**2)
            r_energy = np.sum(hrir[1]**2)
            ild_db = 10 * np.log10((l_energy + 1e-10) / (r_energy + 1e-10))
        
        # Add to validated dictionary
        valid_hrirs_dict[az] = hrir
    
    # In load_hrir(), add this assertion at the end
    for az, hrir in valid_hrirs_dict.items():
        valid_hrirs_dict[az] = hrir.astype(np.float32)
        assert hrir.shape[0] == 2, f"Expected HRIR shape [2, N], got {hrir.shape}"

    print(f"Loaded and validated {len(valid_hrirs_dict)} HRIR azimuth positions")
    return valid_hrirs_dict  # Return the validated dictionary



# Keep this version of apply_hrir_fixed
def apply_hrir_fixed(mono_audio, hrir_l, hrir_r, target_sr=16000, target_length=None):
    """
    Apply HRIR to mono signal to create binaural signal with fixed length
    """
    # Validation checks for HRIRs
    assert hrir_l.ndim == 1 and hrir_r.ndim == 1, "HRIRs must be 1D arrays"
    
    # Use full convolution
    full_l = signal.convolve(mono_audio, hrir_l, mode='full')
    full_r = signal.convolve(mono_audio, hrir_r, mode='full')

    # Use peak normalization to better preserve ILD
    peak = max(np.max(np.abs(full_l)), np.max(np.abs(full_r)), 1e-9)
    binaural_l = full_l / peak * 0.99
    binaural_r = full_r / peak * 0.99
    
    # Stack channels
    binaural = np.vstack((binaural_l, binaural_r))
    
    # Ensure target length if specified
    if target_length is not None:
        if binaural.shape[1] > target_length:
            binaural = binaural[:, :target_length]
        elif binaural.shape[1] < target_length:
            pad_width = ((0, 0), (0, target_length - binaural.shape[1]))
            binaural = np.pad(binaural, pad_width)
    
    return binaural

def create_isotropic_noise(noise_file, hrirs_dict, duration, sr=16000):
    """
    Generate isotropic noise by placing uncorrelated noise at various azimuths
    
    Args:
        noise_file: Path to noise file
        hrirs_dict: Dictionary of HRIRs indexed by azimuth
        duration: Target duration in seconds
        sr: Sample rate (16 kHz as per paper)
    
    Returns:
        binaural_noise: Isotropic noise signal [2, samples]
    """
    try:
        # Load noise file and check sampling rate
        noise, file_sr = librosa.load(noise_file, sr=None, mono=True)
        
        # Resample if needed
        if file_sr != sr:
            noise = librosa.resample(noise, orig_sr=file_sr, target_sr=sr)
        
        # Ensure noise is long enough
        if len(noise) < duration * sr:
            # Repeat noise if needed
            repeats = int(np.ceil((duration * sr) / len(noise)))
            noise = np.tile(noise, repeats)
        
        # Trim noise to target duration
        noise = noise[:int(duration * sr)]
        
        # Initialize output
        target_samples = int(duration * sr)
        binaural_noise = np.zeros((2, target_samples))
        
        # Place uncorrelated noise sources every 5 degrees in azimuthal plane as described in the paper
        azimuths = range(-90, 91, 5)  # -90 to 90 in steps of 5
        available_azimuths = list(hrirs_dict.keys())
        
        for az in azimuths:
            # Find the closest available azimuth
            closest_az = min(available_azimuths, key=lambda x: abs(x - az))
            
            # Get a segment of noise and ensure it's uncorrelated with others
            segment_start = np.random.randint(0, max(1, len(noise) - target_samples))
            segment = noise[segment_start:segment_start + target_samples]
            
            # Pad if segment is too short
            if len(segment) < target_samples:
                segment = np.pad(segment, (0, target_samples - len(segment)))
            
            # Add random phase shift for decorrelation
            segment = add_random_phase(segment)
            
            # Apply HRIR for this azimuth
            hrir = hrirs_dict[closest_az]
            
            # Use a shorter convolution to avoid length issues
            binaural_segment = apply_hrir_fixed(segment, hrir[0], hrir[1], target_sr=sr, target_length=target_samples)
            
            # Check shapes before adding
            if binaural_segment.shape[1] != target_samples:
                # Resize to match target length
                if binaural_segment.shape[1] > target_samples:
                    binaural_segment = binaural_segment[:, :target_samples]
                else:
                    # Pad if too short
                    pad_width = ((0, 0), (0, target_samples - binaural_segment.shape[1]))
                    binaural_segment = np.pad(binaural_segment, pad_width)
            
            # Add to total noise with scaling to avoid clipping
            num_azimuths = len(list(set(available_azimuths) & set(range(-90, 91, 5))))
            scale = 1.0 / max(1, num_azimuths)
            binaural_noise += binaural_segment * scale
        
        return binaural_noise
    except Exception as e:
        print(f"Error creating isotropic noise: {e}")
        # Return fallback noise - white noise with some spatial cues
        fallback_noise = np.random.normal(0, 0.01, (2, int(duration * sr)))
        # Add basic stereo effect
        fallback_noise[0] *= 1.2  # Left channel slightly louder
        fallback_noise[1] *= 0.8  # Right channel slightly quieter
        return fallback_noise

def add_random_phase(signal):
    """Add random phase to signal to decorrelate it"""
    try:
        # FFT
        spec = np.fft.rfft(signal)
        
        # Random phase
        phase = np.exp(1j * np.random.uniform(0, 2*np.pi, len(spec)))
        
        # Apply phase and inverse FFT
        return np.fft.irfft(spec * phase, len(signal))
    except Exception as e:
        print(f"Error in add_random_phase: {e}")
        # Return original signal as fallback
        return signal

def mix_speech_and_noise(speech, noise, target_snr):
    """Mix speech and noise at specified SNR"""
    try:
        # Calculate speech and noise power
        speech_power = np.mean(speech**2)
        noise_power = np.mean(noise**2)
        
        # Handle edge cases
        if speech_power <= 0 or noise_power <= 0:
            print("Warning: Zero power detected in speech or noise")
            return speech  # Return clean speech as fallback
        
        # Calculate scaling factor for noise
        scaling_factor = np.sqrt(speech_power / (noise_power * (10**(target_snr/10))))
        
        # Scale noise and add to speech
        scaled_noise = noise * scaling_factor
        
        # Ensure same length
        min_len = min(speech.shape[1], scaled_noise.shape[1])
        speech = speech[:, :min_len]
        scaled_noise = scaled_noise[:, :min_len]
        
        # Mix
        noisy_speech = speech + scaled_noise
        
        return noisy_speech
    except Exception as e:
        print(f"Error in mix_speech_and_noise: {e}")
        # Return clean speech as fallback
        return speech

def find_wav_files(directory, recursive=True):
    """
    Find all WAV files in a directory with proper handling for TIMIT structure
    
    Args:
        directory: Root directory to search
        recursive: Whether to search recursively in subdirectories
    
    Returns:
        List of Path objects for WAV files
    """
    wav_files = []
    
    # Check if this might be TIMIT structure
    is_timit_structure = False
    test_dir = os.path.join(directory, 'test')
    train_dir = os.path.join(directory, 'train')
    
    if os.path.isdir(test_dir) and os.path.isdir(train_dir):
        is_timit_structure = True
        print("Detected TIMIT directory structure")
        
        # Recursively find all WAV files in test and train directories
        for subdir in ['test', 'train']:
            for root, dirs, files in os.walk(os.path.join(directory, subdir)):
                for file in files:
                    if file.lower().endswith('.wav'):
                        wav_files.append(Path(os.path.join(root, file)))
    else:
        # Regular directory search
        if recursive:
            wav_files = list(Path(directory).rglob('*.wav'))
        else:
            wav_files = list(Path(directory).glob('*.wav'))
    
    print(f"Found {len(wav_files)} WAV files in {directory}")
    return wav_files

def prepare_dataset(clean_dir, noise_dir, hrir_path, output_base_dir, format='wav',
                   split_ratio=[0.7, 0.15, 0.15], dataset_type='vctk', 
                   use_snr_subdirs=True):
    """
    Prepare binaural speech dataset with noise for BCCTN model
    
    Args:
        clean_dir: Directory containing clean speech files (VCTK/TIMIT)
        noise_dir: Directory containing noise files (NOISEX92)
        hrir_path: Path to HRIR files
        output_base_dir: Base directory for output
        format: Format of HRIR files ('wav' or 'mat')
        split_ratio: Train/validation/test split ratio
        dataset_type: 'vctk' for training or 'timit' for unmatched testing
        use_snr_subdirs: Whether to use SNR-specific subdirectories for TIMIT
    """
    # Target sampling rate as specified in the paper
    target_sr = 16000
    
    # Define SNR levels as used in the paper for both VCTK and TIMIT
    snr_levels = [-6, -3, 0, 3, 6, 9, 12, 15]
    
    # Load HRIRs
    print("Loading HRIRs...")
    hrirs_dict = load_hrir(hrir_path, format, target_sr=target_sr)
    
    # Create output directories
    if dataset_type == 'vctk':
        output_dirs = {
            'clean_train': os.path.join(output_base_dir, 'clean_trainset'),
            'clean_val': os.path.join(output_base_dir, 'clean_valset'),
            'clean_test': os.path.join(output_base_dir, 'clean_testset'),
            'noisy_train': os.path.join(output_base_dir, 'noisy_trainset'),
            'noisy_val': os.path.join(output_base_dir, 'noisy_valset'),
            'noisy_test': os.path.join(output_base_dir, 'noisy_testset')
        }
        
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Create SNR subdirectories for VCTK test set for easier evaluation
        if use_snr_subdirs:
            for snr in snr_levels:
                snr_dir = os.path.join(output_dirs['noisy_test'], f'snr_{snr}dB')
                os.makedirs(snr_dir, exist_ok=True)
    else:  # timit - create test dirs with SNR subdirectories
        output_dirs = {
            'clean_test': os.path.join(output_base_dir, 'clean_testset_timit'),
            'noisy_test': os.path.join(output_base_dir, 'noisy_testset_timit')
        }
        
        # Create main directories
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # For TIMIT test, create SNR-specific directories if needed
        if use_snr_subdirs:
            for snr in snr_levels:
                snr_dir = os.path.join(output_dirs['noisy_test'], f'snr_{snr}dB')
                os.makedirs(snr_dir, exist_ok=True)
    
    # Get all clean speech files with special handling for TIMIT
    clean_files = find_wav_files(clean_dir)
    
    if len(clean_files) == 0:
        print(f"Error: No WAV files found in {clean_dir}. Cannot continue.")
        return
    
    # Get all noise files (for both VCTK and TIMIT)
    noise_files = find_wav_files(noise_dir)
    
    # Create a white noise file if no noise files are found
    if len(noise_files) == 0:
        print(f"Error: No noise files found in {noise_dir}. Using white noise instead.")
        white_noise = np.random.normal(0, 0.1, target_sr * 5)  # 5 seconds of white noise
        white_noise_path = os.path.join(output_base_dir, 'white_noise.wav')
        sf.write(white_noise_path, white_noise, target_sr)
        noise_files = [Path(white_noise_path)]
    
    # Process TIMIT separately
    if dataset_type == 'timit':
        process_timit_dataset(clean_files, noise_files, hrirs_dict, output_dirs, 
                              snr_levels, target_sr, use_snr_subdirs)
        return
    
    # VCTK processing starts here
    random.shuffle(clean_files)
    
    # Split into train/val/test
    n_files = len(clean_files)
    n_train = int(n_files * split_ratio[0])
    n_val = int(n_files * split_ratio[1])
    
    train_files = clean_files[:n_train]
    val_files = clean_files[n_train:n_train+n_val]
    test_files = clean_files[n_train+n_val:]
    
    # Process VCTK test set with fixed SNR levels (like TIMIT)
    process_vctk_test_set(test_files, noise_files, hrirs_dict, output_dirs,
                         snr_levels, target_sr, use_snr_subdirs)
    
    # Process train and val sets with random SNRs in ranges
    process_vctk_train_val_sets(train_files, val_files, noise_files, hrirs_dict, 
                                output_dirs, target_sr)

def process_timit_dataset(clean_files, noise_files, hrirs_dict, output_dirs, 
                         snr_levels, target_sr, use_snr_subdirs):
    """Process TIMIT dataset with consistent azimuths across SNR levels"""
    
    print(f"Processing TIMIT dataset with consistent azimuths across SNR levels...")
    
    for i, file_path in enumerate(tqdm.tqdm(clean_files)):
        try:
            # Load and process speech
            speech, file_sr = librosa.load(str(file_path), sr=None, mono=True)
            
            # Skip files that are too short (less than 0.5 seconds)
            if len(speech) / file_sr < 0.5:
                print(f"Skipping {file_path}: too short ({len(speech) / file_sr:.2f} seconds)")
                continue
                
            # Resample if necessary
            if file_sr != target_sr:
                speech = librosa.resample(speech, orig_sr=file_sr, target_sr=target_sr)
            
            # Ensure 2 seconds duration
            target_len = 2 * target_sr
            
            # Handle short audio properly
            if len(speech) < target_len:
                speech = np.pad(speech, (0, target_len - len(speech)))
            else:
                try:
                    start = np.random.randint(0, max(1, len(speech) - target_len))
                    speech = speech[start:start + target_len]
                except Exception as e:
                    print(f"Error trimming audio: {e}. Padding instead.")
                    speech = speech[:target_len]
                    if len(speech) < target_len:
                        speech = np.pad(speech, (0, target_len - len(speech)))
            
            # Ensure exact target length
            if len(speech) != target_len:
                speech = speech[:target_len]
                if len(speech) < target_len:
                    speech = np.pad(speech, (0, target_len - len(speech)))
            
            # Choose ONE random azimuth to use across all SNR levels
            available_azimuths = [az for az in hrirs_dict.keys() if -60 <= az <= 60]
            if not available_azimuths:
                available_azimuths = list(hrirs_dict.keys())
            azimuth = random.choice(available_azimuths)
            
            # Extract IDs
            speaker_id = file_path.parent.name
            sentence_id = file_path.stem
            base_id = f"{speaker_id}_{sentence_id}"
            
            # Create binaural clean speech just once
            hrir = hrirs_dict[azimuth]
            binaural_speech = apply_hrir_fixed(speech, hrir[0], hrir[1], 
                                        target_sr=target_sr, target_length=target_len)
            
            # Save clean speech once
            clean_filename = f"{base_id}_az{azimuth}.wav"
            clean_output_path = os.path.join(output_dirs['clean_test'], clean_filename)
            sf.write(clean_output_path, binaural_speech.T, target_sr)
            
            # For each SNR level, create a corresponding noisy version
            for snr in snr_levels:
                # Choose noise
                noise_file = random.choice(noise_files)
                
                # Create noise
                binaural_noise = create_isotropic_noise(
                    str(noise_file), hrirs_dict, duration=2, sr=target_sr)
                
                # Mix at this specific SNR
                noisy_speech = mix_speech_and_noise(binaural_speech, binaural_noise, snr)
                
                # Save to appropriate directory
                # noisy_filename = f"{base_id}_az{azimuth}_snr{snr}.wav"
                # Always use this format:
                noisy_filename = f"{base_id}_az{azimuth}_snr{snr:+.1f}.wav"
                if use_snr_subdirs:
                    # Use SNR subdirectories
                    snr_dir = os.path.join(output_dirs['noisy_test'], f'snr_{snr}dB')
                    noisy_output_path = os.path.join(snr_dir, noisy_filename)
                else:
                    # Use flat structure
                    noisy_output_path = os.path.join(output_dirs['noisy_test'], noisy_filename)
                
                sf.write(noisy_output_path, noisy_speech.T, target_sr)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print("TIMIT dataset processing complete!")

def process_vctk_test_set(test_files, noise_files, hrirs_dict, output_dirs,
                         snr_levels, target_sr, use_snr_subdirs):
    """Process VCTK test set with fixed SNR levels for consistent evaluation"""
    
    print(f"Processing VCTK test set with fixed SNR levels for paper-style evaluation...")
    
    for i, file_path in enumerate(tqdm.tqdm(test_files)):
        try:
            # Load mono clean speech
            speech, file_sr = librosa.load(str(file_path), sr=None, mono=True)
            
            # Skip files that are too short
            if len(speech) / file_sr < 0.5:
                print(f"Skipping {file_path}: too short ({len(speech) / file_sr:.2f} seconds)")
                continue
            
            # Resample if necessary
            if file_sr != target_sr:
                speech = librosa.resample(speech, orig_sr=file_sr, target_sr=target_sr)
            
            # Ensure 2 seconds duration
            target_len = 2 * target_sr
            
            # Handle short audio properly
            if len(speech) < target_len:
                speech = np.pad(speech, (0, target_len - len(speech)))
            else:
                try:
                    start = np.random.randint(0, max(1, len(speech) - target_len))
                    speech = speech[start:start + target_len]
                except Exception as e:
                    print(f"Error trimming audio: {e}. Padding instead.")
                    speech = speech[:target_len]
                    if len(speech) < target_len:
                        speech = np.pad(speech, (0, target_len - len(speech)))
            
            # Ensure exact target length
            if len(speech) != target_len:
                speech = speech[:target_len]
                if len(speech) < target_len:
                    speech = np.pad(speech, (0, target_len - len(speech)))
            
            # Choose ONE random azimuth to use across all SNR levels
            available_azimuths = [az for az in hrirs_dict.keys() if -60 <= az <= 60]
            if not available_azimuths:
                available_azimuths = list(hrirs_dict.keys())
            azimuth = random.choice(available_azimuths)
            
            # Extract IDs
            speaker_id = file_path.parent.name
            sentence_id = file_path.stem
            base_id = f"{speaker_id}_{sentence_id}"
            
            # Create binaural clean speech just once
            hrir = hrirs_dict[azimuth]
            binaural_speech = apply_hrir_fixed(speech, hrir[0], hrir[1], 
                                        target_sr=target_sr, target_length=target_len)
            
            # Save clean speech once
            clean_filename = f"{base_id}_az{azimuth}.wav"
            clean_output_path = os.path.join(output_dirs['clean_test'], clean_filename)
            sf.write(clean_output_path, binaural_speech.T, target_sr)
            
            # For each SNR level, create a corresponding noisy version
            for snr in snr_levels:
                # Choose noise
                noise_file = random.choice(noise_files)
                
                # Create noise
                binaural_noise = create_isotropic_noise(
                    str(noise_file), hrirs_dict, duration=2, sr=target_sr)
                
                # Mix at this specific SNR
                noisy_speech = mix_speech_and_noise(binaural_speech, binaural_noise, snr)
                
                # Save to appropriate directory
                noisy_filename = f"{base_id}_az{azimuth}_snr{snr:+.1f}.wav"
                if use_snr_subdirs:
                    # Use SNR subdirectories
                    snr_dir = os.path.join(output_dirs['noisy_test'], f'snr_{snr}dB')
                    noisy_output_path = os.path.join(snr_dir, noisy_filename)
                else:
                    # Use flat structure
                    noisy_output_path = os.path.join(output_dirs['noisy_test'], noisy_filename)
                
                sf.write(noisy_output_path, noisy_speech.T, target_sr)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print("VCTK test set processing complete!")

def process_vctk_train_val_sets(train_files, val_files, noise_files, hrirs_dict, 
                              output_dirs, target_sr):
    """Process VCTK train and validation sets with random SNRs"""
    
    print(f"Processing VCTK train and validation sets...")
    
    # SNR range for training/validation (-7 to 16 dB as per paper)
    min_snr = -7
    max_snr = 16
    
    # Process training files
    process_files_with_random_snr(train_files, noise_files, hrirs_dict, 
                                 output_dirs['clean_train'], output_dirs['noisy_train'],
                                 min_snr, max_snr, target_sr)
    
    # Process validation files
    process_files_with_random_snr(val_files, noise_files, hrirs_dict, 
                                 output_dirs['clean_val'], output_dirs['noisy_val'],
                                 min_snr, max_snr, target_sr)
    
    print("VCTK train and validation sets processing complete!")

def process_files_with_random_snr(files, noise_files, hrirs_dict, clean_output_dir, 
                                 noisy_output_dir, min_snr, max_snr, target_sr):
    """Process a set of files with random SNRs in the given range"""
    
    for i, file_path in enumerate(tqdm.tqdm(files)):
        try:
            # Load mono clean speech
            speech, file_sr = librosa.load(str(file_path), sr=None, mono=True)
            
            # Skip files that are too short (less than 0.5 seconds)
            if len(speech) / file_sr < 0.5:
                print(f"Skipping {file_path}: too short ({len(speech) / file_sr:.2f} seconds)")
                continue
                
            # Resample if necessary
            if file_sr != target_sr:
                speech = librosa.resample(speech, orig_sr=file_sr, target_sr=target_sr)
            
            # Ensure 2 seconds duration
            target_len = 2 * target_sr
            
            # Handle short audio properly
            if len(speech) < target_len:
                speech = np.pad(speech, (0, target_len - len(speech)))
            else:
                try:
                    start = np.random.randint(0, max(1, len(speech) - target_len))
                    speech = speech[start:start + target_len]
                except Exception as e:
                    print(f"Error trimming audio: {e}. Padding instead.")
                    speech = speech[:target_len]
                    if len(speech) < target_len:
                        speech = np.pad(speech, (0, target_len - len(speech)))
            
            # Ensure exact target length
            if len(speech) != target_len:
                speech = speech[:target_len]
                if len(speech) < target_len:
                    speech = np.pad(speech, (0, target_len - len(speech)))
            
            # Choose random azimuth
            available_azimuths = [az for az in hrirs_dict.keys() if -60 <= az <= 60]
            if not available_azimuths:
                available_azimuths = list(hrirs_dict.keys())
            azimuth = random.choice(available_azimuths)
            
            # Extract IDs
            speaker_id = file_path.parent.name
            sentence_id = file_path.stem
            base_id = f"{speaker_id}_{sentence_id}"
            
            # Create binaural clean speech
            hrir = hrirs_dict[azimuth]
            binaural_speech = apply_hrir_fixed(speech, hrir[0], hrir[1], 
                                        target_sr=target_sr, target_length=target_len)
            
            # Save clean speech
            clean_filename = f"{base_id}_az{azimuth}.wav"
            clean_output_path = os.path.join(clean_output_dir, clean_filename)
            sf.write(clean_output_path, binaural_speech.T, target_sr)
            
            # Choose random SNR
            snr = round(random.uniform(min_snr, max_snr), 1)
            
            # Choose noise
            noise_file = random.choice(noise_files)
            
            # Create noise
            binaural_noise = create_isotropic_noise(
                str(noise_file), hrirs_dict, duration=2, sr=target_sr)
            
            # Mix at this specific SNR
            noisy_speech = mix_speech_and_noise(binaural_speech, binaural_noise, snr)
            
            # Save noisy speech
            noisy_filename = f"{base_id}_az{azimuth}_snr{snr:+.1f}.wav"
            noisy_output_path = os.path.join(noisy_output_dir, noisy_filename)
            sf.write(noisy_output_path, noisy_speech.T, target_sr)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Prepare binaural speech dataset for BCCTN")
    parser.add_argument("--clean_dir", required=True, help="Directory containing clean speech files")
    parser.add_argument("--noise_dir", required=True, help="Directory containing noise files")
    parser.add_argument("--hrir_path", required=True, help="Path to HRIR files or directory")
    parser.add_argument("--output_dir", required=True, help="Base directory for output")
    parser.add_argument("--hrir_format", default="wav", choices=["wav", "mat"], help="Format of HRIR files")
    parser.add_argument("--dataset_type", default="vctk", choices=["vctk", "timit"], help="Dataset type")
    parser.add_argument("--use_snr_subdirs", default=True, type=bool, help="Use SNR-specific subdirectories")
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.clean_dir,
        args.noise_dir,
        args.hrir_path,
        args.output_dir,
        args.hrir_format,
        dataset_type=args.dataset_type,
        use_snr_subdirs=args.use_snr_subdirs
    )

if __name__ == "__main__":
    main()