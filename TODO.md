# TODO 
1. SegSNR issue
2. change vctk test dataset is spilt to snr
3. check if the model setup is the same as paper
4. run eval in fewer num samples
5. check encoder sharing-weight or not



current package:
name: bcctn
channels:
  - nvidia          # 提供 pytorch‑cuda 12.x
  - pytorch         # 官方 PyTorch wheel
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.3.*
  - pytorch-cuda=12.1          # 會自動拉取 cudnn/cuda-nvcc 等相容版本
  - torchaudio=2.3.*
  - lightning=2.2.*            # (= pytorch-lightning 2.2)
  - hydra-core=1.3.2
  - omegaconf=2.3.0
  - librosa=0.8.1
  - numpy=1.22.3
  - scipy=1.11.*
  - matplotlib=3.5.1
  - ipython
  - flake8
  - pip
  - pip:
      - torch_stoi==0.1.2      # 目前僅 pip 發佈

下面先給 **1 秒、16 kHz 立體聲輸入**（T = 16 000 samples → 157 frames）的 **完整逐層統計**，再談硬體瓶頸。

> *FLOPs＝MAC×2，ComplexConv 以 2×RealConv 權重估算；注意力 softmax 開銷忽略，FFT 取 radix-2.*

| #                  | Layer (kernel/stride)     | Out-shape *(B=1)* |      Params |       FLOPs |
| ------------------ | ------------------------- | ----------------- | ----------: | ----------: |
| **0**              | **STFT (512, 400, 100)**  | (2, 257, 157)     |           0 |    **14 M** |
| **Enc-1**          | CConv2d 2→16 (5×1 /2,1)   | (16, 129, 157)    |   3.2 × 10² |        65 M |
| **Enc-2**          | CConv2d 16→32             | (32, 65, 157)     |       5.1 k |       129 M |
| **Enc-3**          | CConv2d 32→64             | (64, 33, 157)     |   2.0 × 10⁴ |       258 M |
| **Enc-4**          | CConv2d 64→128            | (128, 17, 157)    |   8.2 × 10⁴ |       514 M |
| **Enc-5**          | CConv2d 128→256           | (256, 9, 157)     |   3.3 × 10⁵ |      1.03 G |
| **Enc-6**          | CConv2d 256→256           | (256, 5, 157)     |   6.6 × 10⁵ |      1.36 G |
| **Flatten**        | reshape → (T=157, E=1024) | —                 |           — |           — |
| **Proj-in**        | Linear 1024→512           | (157, 512)        |  5.24 × 10⁵ |       161 M |
| **MHA (32 heads)** | QKV+Attn+Out              | (157, 512)        |      1.05 M |       547 M |
| **Proj-out**       | Linear 512→1024           | (157, 1024)       |  5.24 × 10⁵ |       161 M |
| **Split**          | → L/R (256, 5, 157)×2     | —                 |           — |           — |
| **Dec-1**          | CConvT 256→256            | (256, 9, 157)     |   6.6 × 10⁵ |      1.35 G |
| **Dec-2**          | CConvT 256→128            | (128, 17, 157)    |   3.3 × 10⁵ |      1.03 G |
| **Dec-3**          | CConvT 128→64             | (64, 33, 157)     |   8.2 × 10⁴ |       514 M |
| **Dec-4**          | CConvT 64→32              | (32, 65, 157)     |   2.0 × 10⁴ |       258 M |
| **Dec-5**          | CConvT 32→16              | (16, 129, 157)    |       5.1 k |       129 M |
| **Dec-6**          | CConvT 16→2               | (2, 257, 157)     |   3.2 × 10² |        65 M |
| **CRM mask**       | elem-wise mult            | (2, 257, 157)     |           0 |        41 M |
| **iSTFT**          | overlap-add               | (2, T)            |           0 |        14 M |
| **TOTAL**          |                           |                   | **≈ 4.3 M** | **≈ 4.6 G** |

*源碼：`model.py`, `binaural_attention_model.py`*&#x20;

---

### 主要硬體瓶頸 & 建議

| 模組                                     | 原因                                                              | ASIC/FPGA 對策                                                                                                          |
| -------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Encoder / Decoder CConv(Transpose)** | 5×1 kernel → **80 %+ FLOPs**；複數運算＝4×實數 MAC；skip-cat 造成高 SRAM 帶寬 | *Weight-stationary* PE 陣列；把 real/imag 交錯壓縮，8-bit 固定點 (FP8、INT8)；行列循環 buffer 以 2-line reuse, 避免 DDR ping-pong          |
| **MHA**                                | ① QKV 乘加大矩陣<br>② `S×S` 注意力 (157²) 需兩次 buffer                    | 塊狀或 **block-sparse attention**；Flash-attention-style on-chip compute; 16-bit (bfloat16) reduce power；若**流式**應用可裁剪序列長度 |
| **STFT / iSTFT**                       | 512-pt FFT 每 frame 4608 FLOPs，但頻繁；complex twiddle               | 專用 **FFT/IFFT IP**；將 Win/Hop 設定烤死，用硬浮點或定點 RMS CORE                                                                    |
| **Intermediate SRAM**                  | deepest feature (L/R 256×5×157) ≈ 0.32 MB，需要與 skip features 并行  | 雙埠 SRAM + **on-chip compression (u-law)**；或分段 decode 以降低峰值                                                            |
| **Param 存儲**                           | 4.3 M weights (≈ 17 MB FP32)                                    | 8-bit 重訓, Delta-encoding, BRAM 或 eNVM                                                                                 |
| **複數 BN / PReLU**                      | 逐通道參數 + complex ops                                             | 實部/虛部分開 + LUT，或改用Scale-Shift-ReLU                                                                                     |

> **關鍵瓶頸**＝**(1) 複數 5×1 Conv 帶寬/算力，(2) MHA 的 QKV 大矩陣 & S² 記憶體，(3) STFT 連續 FFT。**
> 優先下手：先把 Conv + FFT 做成共享 **MAC-DSP + Butterfly** 的流水硬核，再視情況引入稀疏注意力或序列裁剪。若目標功耗 < 1 mW（聽力輔具場景），8-bit 定點 + clock gating + SRAM banking 是必要路徑。

