import os
import time
import torch
import numpy as np
import soundfile as sf
import librosa
import argparse
from pathlib import Path

# Import BCCTN-specific modules
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from DCNN.trainer import DCNNLightningModule

def benchmark_model(model_checkpoint, input_file, output_dir, num_runs=5, force_cpu=True):
    """
    Benchmark the BCCTN model on a single audio file.
    Measures processing time on CPU for a single pattern (typically 2 seconds).
    
    Args:
        model_checkpoint: Path to model checkpoint
        input_file: Path to input audio file (noisy binaural)
        output_dir: Directory to save enhanced output and results
        num_runs: Number of runs for averaging timing results
        force_cpu: Whether to force CPU execution even if GPU is available
    """
    print(f"Starting benchmark of BCCTN model")
    print(f"- Using checkpoint: {model_checkpoint}")
    print(f"- Input file: {input_file}")
    print(f"- Force CPU: {force_cpu}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device (force CPU if requested)
    if force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU execution")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize(config_path="config")
    config = compose(config_name="config")
    
    # Load model
    print("Loading model...")
    model = DCNNLightningModule(config)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    print("Model loaded successfully")
    
    # Load input audio
    print(f"Loading input audio: {input_file}")
    noisy_data, sr = sf.read(input_file)
    
    # Verify file is binaural
    if len(noisy_data.shape) == 1 or noisy_data.shape[1] < 2:
        raise ValueError(f"Input file must be binaural (stereo) audio. Found shape: {noisy_data.shape}")
    
    # Get duration
    duration = len(noisy_data) / sr
    print(f"Audio: {duration:.2f} seconds, {sr} Hz")
    
    # Trim to exactly 2 seconds if longer
    if duration > 2.0:
        print(f"Trimming audio to exactly 2 seconds")
        samples_2sec = int(2.0 * sr)
        noisy_data = noisy_data[:samples_2sec]
    
    # Pad to 2 seconds if shorter
    elif duration < 2.0:
        print(f"Padding audio to 2 seconds")
        samples_2sec = int(2.0 * sr)
        noisy_data = np.pad(noisy_data, ((0, samples_2sec - len(noisy_data)), (0, 0)))
    
    # Create input tensor from audio data [1, 2, samples]
    input_tensor = torch.tensor(noisy_data.T, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Warm-up run 
    print("Performing warm-up inference...")
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Benchmark multiple runs
    print(f"Running benchmark with {num_runs} iterations...")
    times = []
    
    for i in range(num_runs):
        # Clear cache if using CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            enhanced_data = model(input_tensor)
        end_time = time.time()
        
        # Calculate time
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"Run {i+1}/{num_runs}: {elapsed_time:.4f} seconds")
    
    # Process timing results
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Save enhanced audio from final run
    enhanced_data = enhanced_data.cpu().numpy()
    enhanced_audio = enhanced_data[0].T  # Convert back to [samples, channels]
    sf.write(os.path.join(output_dir, "enhanced_output.wav"), enhanced_audio, sr)
    
    # Save results
    results = {
        "device": str(device),
        "model": "BCCTN (Binaural Complex Convolutional Transformer Network)",
        "audio_duration": 2.0,  # Fixed to 2 seconds
        "sampling_rate": sr,
        "average_time": avg_time,
        "std_dev": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "realtime_factor": 2.0 / avg_time
    }
    
    # Print results
    print("\n===== BENCHMARK RESULTS =====")
    print(f"Device: {results['device']}")
    print(f"Audio duration: {results['audio_duration']:.2f} seconds")
    print(f"Average processing time: {results['average_time']:.4f} seconds")
    print(f"Standard deviation: {results['std_dev']:.4f} seconds")
    print(f"Min/Max time: {results['min_time']:.4f} / {results['max_time']:.4f} seconds")
    print(f"Realtime factor: {results['realtime_factor']:.4f}x")
    print(f"Required for realtime: {results['realtime_factor'] >= 1.0}")
    
    # Save results to file
    with open(os.path.join(output_dir, "benchmark_results.txt"), "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BCCTN model on CPU")
    parser.add_argument("--model_checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input_file", required=True, help="Path to input audio file (noisy binaural)")
    parser.add_argument("--output_dir", default="./benchmark_results", help="Directory to save results")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark runs to average")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available (default: force CPU)")
    
    args = parser.parse_args()
    
    benchmark_model(
        args.model_checkpoint,
        args.input_file,
        args.output_dir,
        num_runs=args.num_runs,
        force_cpu=not args.use_gpu
    )
