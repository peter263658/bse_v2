import os
import torch
import numpy as np
from collections import OrderedDict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Import model modules
from DCNN.trainer import DCNNLightningModule

def analyze_model(model_checkpoint_path):
    """
    Load model from checkpoint and analyze its architecture and parameters
    """
    print("\n" + "="*50)
    print("Loading model from checkpoint:", model_checkpoint_path)
    print("="*50 + "\n")
    
    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize(config_path="config")
    config = compose(config_name="config")
    
    # Load model
    device = torch.device("cpu")
    model = DCNNLightningModule(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Get the actual model from the lightning module
    bcctn_model = model.model
    
    # 1. Print total parameters
    total_params = sum(p.numel() for p in bcctn_model.parameters())
    trainable_params = sum(p.numel() for p in bcctn_model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\n" + "="*50)
    
    # 2. Print layer-by-layer details
    print("MODEL ARCHITECTURE BREAKDOWN")
    print("="*50)
    
    # STFT/ISTFT layers
    stft_params = sum(p.numel() for p in bcctn_model.stft.parameters())
    istft_params = sum(p.numel() for p in bcctn_model.istft.parameters())
    print(f"STFT Layer: {stft_params:,} parameters")
    print(f"ISTFT Layer: {istft_params:,} parameters")
    
    # Encoder layers
    print("\nENCODER LAYERS:")
    total_encoder_params = 0
    for i, layer in enumerate(bcctn_model.encoder.model):
        params = sum(p.numel() for p in layer.parameters())
        total_encoder_params += params
        in_ch = bcctn_model.kernel_num[i]//2
        out_ch = bcctn_model.kernel_num[i+1]//2
        print(f"  Encoder Block {i+1}: {in_ch}->{out_ch} channels, {params:,} parameters")
        
        # Breakdown of complex layers
        conv_params = sum(p.numel() for name, p in layer.named_parameters() if 'conv' in name)
        bn_params = sum(p.numel() for name, p in layer.named_parameters() if 'bn' in name)
        print(f"    - ComplexConv2d: {conv_params:,} parameters")
        print(f"    - ComplexBatchNorm: {bn_params:,} parameters")
        
    print(f"Total Encoder Parameters: {total_encoder_params:,}")
    
    # Transformer block
    print("\nTRANSFORMER BLOCK:")
    flatten_size = bcctn_model.mattn.input_proj.in_features
    embed_dim = bcctn_model.mattn.mattn.embed_dim
    num_heads = bcctn_model.mattn.mattn.num_heads
    
    input_proj_params = sum(p.numel() for p in bcctn_model.mattn.input_proj.parameters())
    attn_params = sum(p.numel() for p in bcctn_model.mattn.mattn.parameters())
    output_proj_params = sum(p.numel() for p in bcctn_model.mattn.output_proj.parameters())
    
    total_transformer_params = input_proj_params + attn_params + output_proj_params
    
    print(f"  Input projection ({flatten_size} -> {embed_dim}): {input_proj_params:,} parameters")
    print(f"  MultiheadAttention (heads={num_heads}): {attn_params:,} parameters")
    print(f"  Output projection ({embed_dim} -> {flatten_size}): {output_proj_params:,} parameters")
    print(f"Total Transformer Parameters: {total_transformer_params:,}")
    
    # Decoder layers
    print("\nDECODER LAYERS:")
    total_decoder_params = 0
    for i, layer in enumerate(bcctn_model.decoder.model):
        params = sum(p.numel() for p in layer.parameters())
        total_decoder_params += params
        idx = len(bcctn_model.kernel_num) - 1 - i
        in_ch = bcctn_model.kernel_num[idx]
        out_ch = bcctn_model.kernel_num[idx-1]//2
        print(f"  Decoder Block {i+1}: {in_ch}->{out_ch} channels, {params:,} parameters")
        
        # Count convtranspose parameters only
        conv_params = sum(p.numel() for name, p in layer.named_parameters() if 'conv' in name)
        print(f"    - ComplexConvTranspose2d: {conv_params:,} parameters")
        
        # If not the last layer, also has BN and PReLU
        if i < len(bcctn_model.decoder.model) - 1:
            bn_params = sum(p.numel() for name, p in layer.named_parameters() if 'bn' in name)
            print(f"    - ComplexBatchNorm: {bn_params:,} parameters")
    
    print(f"Total Decoder Parameters: {total_decoder_params:,}")
    
    # 3. Verify model calculation flow with a dummy input
    print("\n" + "="*50)
    print("MODEL CALCULATION FLOW")
    print("="*50)
    
    # Create a dummy input (2 seconds of audio at 16kHz)
    dummy_input = torch.randn(1, 2, 32000)  # [batch, channels=2 for binaural, samples]
    
    # Enable hooks to track tensor shapes
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.shape
        return hook
    
    # Register hooks
    hooks = []
    
    # STFT
    h = bcctn_model.stft.register_forward_hook(get_activation('stft'))
    hooks.append(h)
    
    # Encoder blocks
    for i, layer in enumerate(bcctn_model.encoder.model):
        h = layer.register_forward_hook(get_activation(f'encoder_{i}'))
        hooks.append(h)
    
    # Transformer
    h = bcctn_model.mattn.register_forward_hook(get_activation('transformer'))
    hooks.append(h)
    
    # Decoder blocks
    for i, layer in enumerate(bcctn_model.decoder.model):
        h = layer.register_forward_hook(get_activation(f'decoder_{i}'))
        hooks.append(h)
    
    # ISTFT
    h = bcctn_model.istft.register_forward_hook(get_activation('istft'))
    hooks.append(h)
    
    # Run forward pass
    with torch.no_grad():
        output = bcctn_model(dummy_input)
    
    # Print shapes throughout the network
    print(f"Input shape: {dummy_input.shape}")
    for name, shape in activation.items():
        print(f"{name} output shape: {shape}")
    print(f"Final output shape: {output.shape}")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return bcctn_model

if __name__ == "__main__":
    # Update path to your checkpoint
    checkpoint_path = "/raid/R12K41024/BCCTN/outputs/2025-05-11/16-34-17/logs/lightning_logs/version_0/checkpoints/epoch=94-step=91960.ckpt"
    model = analyze_model(checkpoint_path)