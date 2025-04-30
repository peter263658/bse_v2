import numpy as np, scipy.signal as sg

# bark weights from VOICEBOX
BARK_W = np.array(
 [0.13,0.26,0.42,0.60,0.78,0.93,1.00,0.97,0.89,0.76,0.62,
  0.50,0.38,0.28,0.22,0.18,0.14,0.11,0.09,0.07,0.06,0.05,0.04])

# band edges in Hz (VOICEBOX table)
BARK_EDGES = np.array(
 [   0,  100,  200,  300,  400,  510,  630,  770,  920, 1080, 1270,
  1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
  9500,12000])

def fwsegsnr(clean, proc, sr=16000):
    frame, hop = int(0.025*sr), int(0.00625*sr)
    win  = sg.windows.hann(frame, False)
    fftL = 512

    def frames(x):
        nfrm = (len(x)-frame)//hop + 1
        idx  = np.arange(frame)[None,:] + hop*np.arange(nfrm)[:,None]
        return x[idx] * win

    # residual noise
    enoise = proc[:len(clean)] - clean
    C, E   = np.fft.rfft(frames(clean),  fftL, axis=1), \
             np.fft.rfft(frames(enoise), fftL, axis=1)

    # Bark-band powers
    edges = np.round(BARK_EDGES*fftL/sr).astype(int)
    Pw_s, Pw_e = np.zeros(C.shape[0]), np.zeros(E.shape[0])
    for b,w in enumerate(BARK_W):
        lo, hi   = edges[b], edges[b+1]
        Pw_s    += w * np.sum(np.abs(C[:,lo:hi])**2, axis=1)
        Pw_e    += w * np.sum(np.abs(E[:,lo:hi])**2, axis=1)

    snr = 10*np.log10(Pw_s / (Pw_e + 1e-12))
    snr = np.clip(snr, -10, 35)            # frame clipping
    return snr.mean()


# create dummy signals
t   = np.random.randn(64000) * 0.1          # fake speech
noi = np.random.randn(64000) * 0.2          # noise
mix = t + noi
enh = t + noi * 0.01                        # 20 dB better

print("noisy :", fwsegsnr(t, mix))          # ≈ -5 dB
print("enh   :", fwsegsnr(t, enh))          # ≈ +14 dB
