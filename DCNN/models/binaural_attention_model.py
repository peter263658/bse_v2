import torch

from DCNN.utils.apply_mask import apply_mask

from .model import DCNN




class BinauralAttentionDCNN(DCNN):

    # def forward(self, inputs):
    #     # batch_size, binaural_channels, time_bins = inputs.shape
    #     cspecs_l = self.stft(inputs[:, 0])
    #     cspecs_r = self.stft(inputs[:, 1])
    #     cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)

    #     # breakpoint()

    #     # encoder_out_l = self.encoder(attention_enc[:, 0, :, :].unsqueeze(1))
    #     # encoder_out_r = self.encoder(attention_enc[:, 1, :, :].unsqueeze(1))

    #     encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
    #     encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
    #     # breakpoint()

    #     # encoder_out = torch.cat((encoder_out_l[-1], encoder_out_r[-1]), dim=1)
    #     # attention_enc = self.attention_enc(encoder_out)
    #     # breakpoint()
    #     # attention_enc = encoder_out_l[-1]*encoder_out_r[-1].conj()
    #     # attention_enc = self.attention(cspecs)
    #     # _, attn_len, _, _ = attention_enc.shape
    #     # encoder_attn_l = attention_enc[:, :attn_len//2, :, :]
    #     # encoder_attn_r = attention_enc[:, attn_len//2:, :, :]
    #     # breakpoint()
    #     # 2. Apply RNN
    #     # x_l_rnn = self.rnn(encoder_out_l[-1])
    #     # x_r_rnn = self.rnn(encoder_out_r[-1])
    #     # breakpoint()
    #     # attention_in = torch.cat((encoder_out_l[-1],encoder_out_r[-1]), dim=1)
        
    #     #
    #     # breakpoint()
    #     # x_l_mattn = self.mattn(encoder_out_l[-1])
    #     # x_r_mattn = self.mattn(encoder_out_r[-1])
    #     # breakpoint()

    #     # x_attn = self.mattn(attention_in)
    #     # rnn_out = torch.cat((x_l_rnn, x_r_rnn), dim=1)
    #     # breakpoint()
    #     # attention_dec = self.attention_enc(rnn_out)

    #     # _, dec_attn_len, _, _ = attention_dec.shape
    #     # decoder_attn_l = attention_dec[:, :dec_attn_len//2, :, :]
    #     # decoder_attn_r = attention_dec[:, dec_attn_len//2:, :, :]
    #     # x_l_mattn = x_attn[:,:128,:,:]
    #     # x_r_mattn = x_attn[:,128:,:,:]
    #     # x_l_mattn = x_attn[:,:64,:,:]
    #     # x_r_mattn = x_attn[:,64:,:,:]


    #     enc_last = torch.cat([encoder_out_l[-1], encoder_out_r[-1]], dim=1)  # C=256
    #     x_attn   = self.mattn(enc_last)                                      # input_size=1024
    #     # 之後再 split 成兩半送兩個 decoder
    #     x_l_mattn, x_r_mattn = x_attn.chunk(2, dim=1)

    #     # x_l_mattn = self.mattn(encoder_out_l[-1])
    #     # x_r_mattn = self.mattn(encoder_out_r[-1])
    #     # 3. Apply decoder
    #     # x_l = self.decoder(x_l_rnn, encoder_out_l)
    #     # x_r = self.decoder(x_r_rnn, encoder_out_r)
    #     # x_l = self.decoder(x_l_mattn, encoder_out_l)
    #     # x_r = self.decoder(x_r_mattn, encoder_out_r)
    #     x_l = self.decoder(x_l_mattn, encoder_out_l)
    #     x_r = self.decoder(x_r_mattn, encoder_out_r)

    #     # 4. Apply mask
    #     out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
    #     out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)

    #     # 5. Invert STFT
    #     out_wav_l = self.istft(out_spec_l)
    #     # breakpoint()
    #     # out_wav_l = torch.squeeze(out_wav_l, 1)
    #     # out_wav_l = torch.clamp_(out_wav_l, -1, 1)
        
    #     out_wav_r = self.istft(out_spec_r)
    #     # out_wav_r = torch.squeeze(out_wav_r, 1)
    #     # out_wav_r = torch.clamp_(out_wav_r, -1, 1)

    #     # breakpoint()
        
    #     out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
       
    #     return out_wav
    # def forward(self, inputs):
    #     # Process left and right channels through STFT
    #     cspecs_l = self.stft(inputs[:, 0])
    #     cspecs_r = self.stft(inputs[:, 1])
        
    #     # Pass through encoders
    #     encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
    #     encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
        
    #     # Concatenate encoder outputs and send through attention
    #     enc_combined = torch.cat([encoder_out_l[-1], encoder_out_r[-1]], dim=1)
    #     attention_out = self.mattn(enc_combined)
        
    #     # Split attention output back for separate decoders
    #     attn_l, attn_r = attention_out.chunk(2, dim=1)
        
    #     # Decode both channels
    #     x_l = self.decoder(attn_l, encoder_out_l)
    #     x_r = self.decoder(attn_r, encoder_out_r)
        
    #     # Apply masks
    #     out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
    #     out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)
        
    #     # Inverse STFT
    #     out_wav_l = self.istft(out_spec_l)
    #     out_wav_r = self.istft(out_spec_r)
        
    #     # Stack for output
    #     out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
        
    #     return out_wav


    def forward(self, inputs):
        # Process left and right channels through STFT
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        
        # Pass through encoders
        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
        
        # Concatenate encoder outputs and send through attention
        enc_combined = torch.cat([encoder_out_l[-1], encoder_out_r[-1]], dim=1)
        attention_out = self.mattn(enc_combined)
        
        # Split attention output back for separate decoders
        attn_l, attn_r = attention_out.chunk(2, dim=1)
        
        # Decode both channels
        x_l = self.decoder(attn_l, encoder_out_l)
        x_r = self.decoder(attn_r, encoder_out_r)
        
        # Apply masks
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)
        
        # Inverse STFT
        out_wav_l = self.istft(out_spec_l)
        out_wav_r = self.istft(out_spec_r)
        
        # Stack for output
        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
        
        return out_wav