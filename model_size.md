==================================================
MODEL ARCHITECTURE BREAKDOWN
==================================================
STFT Layer: 0 parameters
ISTFT Layer: 0 parameters

ENCODER LAYERS:
  Encoder Block 1: 1->8 channels, 130 parameters
    - ComplexConv2d: 96 parameters
    - ComplexBatchNorm: 32 parameters
  Encoder Block 2: 8->16 channels, 1,378 parameters
    - ComplexConv2d: 1,312 parameters
    - ComplexBatchNorm: 64 parameters
  Encoder Block 3: 16->32 channels, 5,314 parameters
    - ComplexConv2d: 5,184 parameters
    - ComplexBatchNorm: 128 parameters
  Encoder Block 4: 32->64 channels, 20,866 parameters
    - ComplexConv2d: 20,608 parameters
    - ComplexBatchNorm: 256 parameters
  Encoder Block 5: 64->128 channels, 82,690 parameters
    - ComplexConv2d: 82,176 parameters
    - ComplexBatchNorm: 512 parameters
  Encoder Block 6: 128->128 channels, 164,610 parameters
    - ComplexConv2d: 164,096 parameters
    - ComplexBatchNorm: 512 parameters
Total Encoder Parameters: 274,988

TRANSFORMER BLOCK:
  Input projection (1024 -> 512): 524,800 parameters
  MultiheadAttention (heads=32): 2,101,248 parameters
  Output projection (512 -> 1024): 525,312 parameters
Total Transformer Parameters: 3,151,360

DECODER LAYERS:
  Decoder Block 1: 256->128 channels, 328,450 parameters
    - ComplexConvTranspose2d: 327,936 parameters
    - ComplexBatchNorm: 512 parameters
  Decoder Block 2: 256->64 channels, 164,226 parameters
    - ComplexConvTranspose2d: 163,968 parameters
    - ComplexBatchNorm: 256 parameters
  Decoder Block 3: 128->32 channels, 41,154 parameters
    - ComplexConvTranspose2d: 41,024 parameters
    - ComplexBatchNorm: 128 parameters
  Decoder Block 4: 64->16 channels, 10,338 parameters
    - ComplexConvTranspose2d: 10,272 parameters
    - ComplexBatchNorm: 64 parameters
  Decoder Block 5: 32->8 channels, 2,610 parameters
    - ComplexConvTranspose2d: 2,576 parameters
    - ComplexBatchNorm: 32 parameters
  Decoder Block 6: 16->1 channels, 162 parameters
    - ComplexConvTranspose2d: 162 parameters
Total Decoder Parameters: 546,940

==================================================
MODEL CALCULATION FLOW
==================================================
Input shape: torch.Size([1, 2, 32000])
stft output shape: torch.Size([1, 256, 321])
encoder_0 output shape: torch.Size([1, 8, 128, 321])
encoder_1 output shape: torch.Size([1, 16, 64, 321])
encoder_2 output shape: torch.Size([1, 32, 32, 321])
encoder_3 output shape: torch.Size([1, 64, 16, 321])
encoder_4 output shape: torch.Size([1, 128, 8, 321])
encoder_5 output shape: torch.Size([1, 128, 4, 321])
transformer output shape: torch.Size([1, 256, 4, 321])
decoder_0 output shape: torch.Size([1, 128, 8, 321])
decoder_1 output shape: torch.Size([1, 64, 16, 321])
decoder_2 output shape: torch.Size([1, 32, 32, 321])
decoder_3 output shape: torch.Size([1, 16, 64, 321])
decoder_4 output shape: torch.Size([1, 8, 128, 321])
decoder_5 output shape: torch.Size([1, 1, 256, 321])
istft output shape: torch.Size([1, 32000])
Final output shape: torch.Size([1, 2, 32000])
