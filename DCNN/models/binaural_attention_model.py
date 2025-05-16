import torch

from DCNN.utils.apply_mask import apply_mask

from .model import DCNN

# class BinauralAttentionDCNN(DCNN):
#     def forward(self, inputs):
#         # Process left and right channels through STFT
#         cspecs_l = self.stft(inputs[:, 0])
#         cspecs_r = self.stft(inputs[:, 1])
        
#         # Pass through encoders
#         encoder_out_l = self.encoder_l(cspecs_l.unsqueeze(1))
#         encoder_out_r = self.encoder_r(cspecs_r.unsqueeze(1))
        
#         # Concatenate encoder outputs and send through attention
#         enc_combined = torch.cat([encoder_out_l[-1], encoder_out_r[-1]], dim=1)
#         attention_out = self.mattn(enc_combined)
        
#         # Split attention output back for separate decoders
#         attn_l, attn_r = attention_out.chunk(2, dim=1)
        
#         # Decode both channels
#         x_l = self.decoder_l(attn_l, encoder_out_l)
#         x_r = self.decoder_r(attn_r, encoder_out_r)
        
#         # Apply masks
#         out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
#         out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)
        
#         # Inverse STFT
#         out_wav_l = self.istft(out_spec_l)
#         out_wav_r = self.istft(out_spec_r)
        
#         # Stack for output
#         out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
        
#         return out_wav

class BinauralAttentionDCNN(DCNN):
    def forward(self, inputs):
        # 1. Get STFT for left and right channels
        cspecs_l = self.stft(inputs[:, 0].unsqueeze(1))  # (B, 1, F, T) complex
        cspecs_r = self.stft(inputs[:, 1].unsqueeze(1))  # (B, 1, F, T) complex
        
        # 2. Process through complex encoder
        encoder_out_l = self.encoder(cspecs_l)
        encoder_out_r = self.encoder(cspecs_r)
        
        # 3. Get last encoder layer outputs
        xl = encoder_out_l[-1]  # (B, C, F, T) complex
        xr = encoder_out_r[-1]  # (B, C, F, T) complex
        
        # 4. Convert to real for attention
        b, c, f, t = xl.shape
        xl_real = torch.view_as_real(xl)         # (B,C,F,T,2)
        xl_real = xl_real.permute(0,1,4,2,3)     # (B,C,2,F,T)
        xl_real = xl_real.reshape(b, 2*c, f, t)  # (B,2C,F,T) real
        
        xr_real = torch.view_as_real(xr)         # (B,C,F,T,2)
        xr_real = xr_real.permute(0,1,4,2,3)     # (B,C,2,F,T)
        xr_real = xr_real.reshape(b, 2*c, f, t)  # (B,2C,F,T) real
        
        # 5. Concatenate for attention
        x_concat = torch.cat([xl_real, xr_real], dim=1)  # (B, 4C, F, T)
        
        # 6. Apply attention
        x_attn = self.mattn(x_concat)  # (B, 4C, F, T)
        
        # 7. Split back for left and right
        split_size = x_attn.size(1) // 2
        xl_attn = x_attn[:, :split_size]        # (B, 2C, F, T)
        xr_attn = x_attn[:, split_size:]        # (B, 2C, F, T)
        
        # 8. Convert back to complex for decoder
        xl_attn = xl_attn.view(b, c, 2, f, t)                  # (B, C, 2, F, T)
        xl_attn = xl_attn.permute(0, 1, 3, 4, 2)               # (B, C, F, T, 2)
        xl_attn = torch.view_as_complex(xl_attn.contiguous())  # (B, C, F, T) complex
        
        xr_attn = xr_attn.view(b, c, 2, f, t)                  # (B, C, 2, F, T)
        xr_attn = xr_attn.permute(0, 1, 3, 4, 2)               # (B, C, F, T, 2)
        xr_attn = torch.view_as_complex(xr_attn.contiguous())  # (B, C, F, T) complex
        
        # 9. Apply decoder with skip connections
        xl_dec = self.decoder(xl_attn, encoder_out_l)
        xr_dec = self.decoder(xr_attn, encoder_out_r)


        # 之後再做 masking
        out_spec_l = apply_mask(xl_dec[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(xr_dec[:, 0], cspecs_r, self.masking_mode)
        # ----------------------------------
        
        # 11. Invert STFT
        out_wav_l = self.istft(out_spec_l)
        out_wav_r = self.istft(out_spec_r)
        
        # 12. Stack for output
        out_wav = torch.stack([out_wav_l.squeeze(1), out_wav_r.squeeze(1)], dim=1)
        
        return out_wav