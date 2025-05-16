import torch
import torch.nn as nn
from DCNN.feature_extractors import IStft, Stft

import DCNN.utils.complexPyTorch.complexLayers as torch_complex
from DCNN.utils.show import show_params, show_model

from DCNN.utils.apply_mask import apply_mask


class DCNN(nn.Module):
    def __init__(
            self,
            rnn_layers=2, rnn_units=128,
            win_len=256, win_inc=100, fft_len=256, win_type='hann',
            masking_mode='E', use_clstm=False,
            kernel_size=5, 
            kernel_num=[16, 32, 64, 128, 256, 256], 
            bidirectional=False, embed_dim=256, num_heads=16, **kwargs
    ):
        super().__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len

        self.rnn_units = rnn_units
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        # self.kernel_num = [2] + kernel_num  # First layer is 2//2=1 complex channel
        self.kernel_num = kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.stft = Stft(self.fft_len, self.win_inc, self.win_len)
        self.istft = IStft(self.fft_len, self.win_inc, self.win_len)
        
        # Complex network components
        self.encoder = Encoder(self.kernel_num, kernel_size)
        self.decoder = Decoder(self.kernel_num, self.kernel_size)
        
        # Real-valued attention
        self.mattn = MultiAttnBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )

    def forward(self, inputs):
        # 1. Get complex STFT
        cspecs = self.stft(inputs)      # complex64, shape (B, Ch, F, T)
                
        # 2. Process through complex encoder (preserving complex values)
        encoder_out = self.encoder(cspecs)
        x = encoder_out[-1]  # Last encoder layer output
        
        # 3. Convert to real for attention
        b, c, f, t = x.shape
        x_real = torch.view_as_real(x)          # (B,C,F,T,2)
        x_real = x_real.permute(0,1,4,2,3)      # (B,C,2,F,T)
        x_real = x_real.reshape(b, 2*c, f, t)   # (B,2C,F,T)  ← real tensor
        
        # 4. Apply attention (real domain)
        x_attn = self.mattn(x_real)
        
        # 5. Convert back to complex for decoder
        x_attn_c = x_attn.view(b, c, 2, f, t)                  # (B,C,2,F,T)
        x_attn_c = x_attn_c.permute(0,1,3,4,2)                # (B,C,F,T,2)
        x_attn_c = torch.view_as_complex(x_attn_c.contiguous())  # (B,C,F,T) complex
        
        # 6. Apply decoder with skip connections
        x_dec = self.decoder(x_attn_c, encoder_out)
        
        # 7. Apply mask
        out_spec = apply_mask(x_dec[:, 0], cspecs, self.masking_mode)
        
        # 8. Invert STFT
        out_wav = self.istft(out_spec)
        
        return out_wav


class Encoder(nn.Module):
    def __init__(self, kernel_num, kernel_size):
        super().__init__()

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        self.model = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            in_c  = 1 if idx == 0 else kernel_num[idx] // 2
            out_c = kernel_num[idx + 1] // 2      # 複數通道
            self.model.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),

                    torch_complex.ComplexConv2d(
                        # self.kernel_num[idx]//2,
                        # self.kernel_num[idx + 1]//2,
                        in_c, out_c,
                        # kernel_size=(self.kernel_size, 2),
                        # stride=(2, 1),
                        # padding=(2, 1)
                        kernel_size=(self.kernel_size, 1),
                        stride=(2, 1),
                        padding=(2, 0)
                    ),
                    torch_complex.NaiveComplexBatchNorm2d(
                        self.kernel_num[idx + 1]//2),
                    torch_complex.ComplexPReLU()
                )
            )

    def forward(self, x):
        output = []
        # 1. Apply encoder
        for idx, layer in enumerate(self.model):
            x = layer(x)
            # x = x[..., :-1] # Experimental
            output.append(x)

        return output


class Decoder(nn.Module):
    def __init__(self, kernel_num, kernel_size):
        super().__init__()

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        self.model = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            block = [
                torch_complex.ComplexConvTranspose2d(
                    self.kernel_num[idx],  # * 2,
                    self.kernel_num[idx - 1]//2,
                    # kernel_size=(self.kernel_size, 2),
                    # stride=(2, 1),
                    # padding=(2, 1),
                    kernel_size=(self.kernel_size, 1),
                    stride=(2, 1),
                    padding=(2, 0),
                    output_padding=(0, 0)
                ),
            ]

            if idx != 1:
                block.append(torch_complex.NaiveComplexBatchNorm2d(
                    self.kernel_num[idx - 1]//2))
                block.append(torch_complex.ComplexPReLU())
            self.model.append(nn.Sequential(*block))

    def _align(self, a, b):
        """中心裁剪，讓 a, b 的 (F, T) 完全一致"""
        F = min(a.size(-2), b.size(-2))
        T = min(a.size(-1), b.size(-1))
        a = a[..., :F, :T]
        b = b[..., :F, :T]
        return a, b

    def forward(self, x, encoder_out):
        for idx, layer in enumerate(self.model):
            enc = encoder_out[-1 - idx]
            x, enc = self._align(x, enc)   # 對齊再 cat
            x = torch.cat([x, enc], 1)
            x = layer(x)
        return x


# class MultiAttnBlock(nn.Module):
#     def __init__(self, input_size, hidden_size, embed_dim=512, num_heads=32, batch_first=True):
#         super().__init__()
#         # Project to embed_dim first (exactly as in paper)
#         self.input_proj = nn.Linear(input_size, embed_dim, dtype=torch.complex64)
#         # Use the right number of heads as in paper
#         self.mattn = torch_complex.ComplexMultiheadAttention(
#             embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)
#         # Project back to original dimension
#         self.output_proj = nn.Linear(embed_dim, input_size, dtype=torch.complex64)

#     def forward(self, x):
#         batch_size, channels, freqs, time_bins = x.shape
#         x = x.flatten(start_dim=1, end_dim=2)
#         x = x.transpose(1, 2)
#         x = self.input_proj(x)
#         x = self.mattn(x)
#         x = self.output_proj(x)
#         x = x.unflatten(-1, (channels, freqs))
#         x = x.movedim(1, -1)
#         return x


class MultiAttnBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=16):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.pre = None   # Will be dynamically created
        self.post = None

    def set_input_size(self, in_dim):
        """Set input size and create projection layers"""
        self.pre = nn.Linear(in_dim, self.mha.embed_dim)
        self.post = nn.Linear(self.mha.embed_dim, in_dim)

    def forward(self, x):  # x: (B, C, F, T) real
        B, C, F, T = x.shape
        xf = x.flatten(1, 2).transpose(1, 2)   # (B, T, C*F)
        
        # Create projection layers if needed
        in_dim = C * F
        if self.pre is None or self.post is None:
            self.set_input_size(in_dim)
            
        # Project, apply attention, and project back
        xf = self.pre(xf)
        attn_out, _ = self.mha(xf, xf, xf)
        xf = self.post(attn_out)
        
        # Reshape back to original dimensions
        xf = xf.transpose(1, 2).reshape(B, C, F, T)
        return xf