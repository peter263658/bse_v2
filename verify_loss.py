#!/usr/bin/env python3
"""
Verify that the loss function implementation matches the paper's description.
This script is meant to be used for verification purposes only, not during training.
"""

import torch
import numpy as np
from DCNN.loss import BinauralLoss

def verify_loss_function():
    """
    Verify that the loss function implementation matches the paper's description:
    L = αL_SNR + βL_STOI + γL_ILD + κL_IPD
    With weights α=1, β=10, γ=1, κ=10
    """
    print("Verifying loss function implementation...")
    
    # Create a binaural loss instance with the paper's weights
    loss_fn = BinauralLoss(
        ild_weight=1,    # γ
        ipd_weight=10,   # κ
        stoi_weight=10,  # β
        snr_loss_weight=1  # α
    )
    
    # Print the weights to verify
    print(f"SNR Loss Weight (α): {loss_fn.snr_loss_weight}")
    print(f"STOI Loss Weight (β): {loss_fn.stoi_weight}")
    print(f"ILD Loss Weight (γ): {loss_fn.ild_weight}")
    print(f"IPD Loss Weight (κ): {loss_fn.ipd_weight}")
    
    # Check if all four components are included in the forward method
    forward_code = str(loss_fn.forward.__code__)
    
    snr_present = "snr_loss" in forward_code or "bin_snr_loss" in forward_code
    stoi_present = "stoi_loss" in forward_code or "bin_stoi_loss" in forward_code
    ild_present = "ild_loss" in forward_code or "bin_ild_loss" in forward_code
    ipd_present = "ipd_loss" in forward_code or "bin_ipd_loss" in forward_code
    
    print("\nLoss components presence check:")
    print(f"SNR Loss (L_SNR) present: {'✓' if snr_present else '✗'}")
    print(f"STOI Loss (L_STOI) present: {'✓' if stoi_present else '✗'}")
    print(f"ILD Loss (L_ILD) present: {'✓' if ild_present else '✗'}")
    print(f"IPD Loss (L_IPD) present: {'✓' if ipd_present else '✗'}")
    
    # Check if speechMask function is used for ILD and IPD (mentioned in paper)
    mask_present = "speechMask" in forward_code or "mask" in forward_code
    print(f"Speech activity mask for ILD/IPD: {'✓' if mask_present else '✗'}")
    
    # Overall verdict
    if all([snr_present, stoi_present, ild_present, ipd_present, mask_present]):
        print("\n✓ Loss function implementation matches the paper's description.")
    else:
        print("\n✗ Loss function implementation does not fully match the paper's description.")
        
        # Detailed recommendations
        if not snr_present:
            print("  - Add SNR Loss component")
        if not stoi_present:
            print("  - Add STOI Loss component")
        if not ild_present:
            print("  - Add ILD Loss component with speech activity masking")
        if not ipd_present:
            print("  - Add IPD Loss component with speech activity masking")
        if not mask_present:
            print("  - Add speech activity masking for ILD and IPD losses")

if __name__ == "__main__":
    verify_loss_function()