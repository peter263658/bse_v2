import torch
import torch.nn as nn


# class Stft(nn.Module):
#     def __init__(self, n_dft=512, hop_size=256, win_length=None,
#                  onesided=True, is_complex=True):

#         super().__init__()

#         self.n_dft = n_dft
#         self.hop_size = hop_size
#         self.win_length = n_dft if win_length is None else win_length
#         self.onesided = onesided
#         self.is_complex = is_complex

#     def forward(self, x: torch.Tensor):
#         "Expected input has shape (batch_size, n_channels, time_steps)"

#         # window = torch.hann_window(self.win_length, device=x.device)
#         window = torch.hamming_window(self.win_length, device=x.device)

#         y = torch.stft(x, self.n_dft, hop_length=self.hop_size,
#                        win_length=self.win_length, onesided=self.onesided,
#                        return_complex=True, window=window, normalized=True)
        
#         y = y[:, 1:] # Remove DC component (f=0hz)

#         # y.shape == (batch_size*channels, time, freqs)

#         if not self.is_complex:
#             y = torch.view_as_real(y)
#             y = y.movedim(-1, 1) # move complex dim to front

#         return y


# class IStft(Stft):

#     def forward(self, x: torch.Tensor):
#         "Expected input has shape (batch_size, n_channels=freq_bins, time_steps)"
#         # window = torch.hann_window(self.win_length, device=x.device)
#         window = torch.hamming_window(self.win_length, device=x.device)

#         y = torch.istft(x, self.n_dft, hop_length=self.hop_size,
#                         win_length=self.win_length, onesided=self.onesided,
#                         window=window,normalized=True)

#         return y

class Stft(nn.Module):
    def __init__(self, n_dft=256, hop_size=100, win_length=256,
                 onesided=True, is_complex=True):
        super().__init__()
        self.n_dft = n_dft
        self.hop_size = hop_size
        self.win_length = win_length if win_length is not None else n_dft
        self.onesided = onesided
        self.is_complex = is_complex

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"
        window = torch.hamming_window(self.win_length, device=x.device)
        
        # Handle multi-channel input properly
        batch_size, n_channels, time_steps = x.shape
        x_flat = x.reshape(-1, time_steps)  # Flatten batch and channels
        
        y = torch.stft(x_flat, self.n_dft, hop_length=self.hop_size,
                      win_length=self.win_length, onesided=self.onesided,
                      return_complex=True, window=window, normalized=True)
        
        # DO NOT remove DC component (keep all bins)
        # Reshape back to include batch and channel dimensions
        y = y.reshape(batch_size, n_channels, y.shape[1], y.shape[2])
        
        return y


class IStft(Stft):
    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, freq_bins, time_steps)"
        window = torch.hamming_window(self.win_length, device=x.device)
        
        # Handle multi-channel input
        batch_size, n_channels, freq_bins, time_bins = x.shape
        x_flat = x.reshape(-1, freq_bins, time_bins)
        
        y = torch.istft(x_flat, self.n_dft, hop_length=self.hop_size,
                       win_length=self.win_length, onesided=self.onesided,
                       window=window, normalized=True)
        
        # Reshape back
        y = y.reshape(batch_size, n_channels, -1)
        
        return y