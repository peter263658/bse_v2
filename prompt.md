

### 1. `feature_extractors.py`

* `IStft.forward()` 要把 `n_fft`, `win_length`, `hop_length` 都改成用 `self.xxx`，有些舊碼是硬寫 512/400，沒改會反變形。
* 一併把 window 函式 (`torch.hann_window`) 的長度改 256。

---

### 2. `model.py`

1. **`hidden_dim` 算法錯**
   你用 `self.fft_len // (2 ** (len(self.kernel_num) + 1))`，
   但 `self.fft_len` 是 256，不是 bins；真正 bins = `fft_len//2+1 = 129`。
   →  建議直接動態抓 `x.size(2)` 來算 flatten，不寫死公式。

```python
# after last conv
b, c, f, t = x.shape
x = x.reshape(b, c * f, t).permute(0, 2, 1)   # (B,T,CF)
self.in_dim = c * f                            # save for later
```

2. **`MultiAttnBlock` 輸入 dim**
   按 256-FFT + 6×stride2，freq bins = 3。
   最深層通道 256 → flatten = 256\*3 = 768 (單耳)，雙耳 concat = 1536。
   你硬寫 512 會立刻 dimension mismatch。
   →  把 `input_size` 換成 `self.in_dim*2`（雙耳），或動態計算。

3. **`x_l_mattn` slicing**
   如果 embed 256，MHA 輸出仍 256，拆左右應該還是一半 128，不是 64。
   `x_attn.shape = (B, T, 256)`

   ```python
   x_l, x_r = torch.split(x_attn, self.embed_dim//2, dim=-1)  # each 128
   ```

4. **屬性命名**
   你 signature 裡叫 `fft_len`，但下面還在用 `self.fft_len` 嗎？
   確定 `self.fft_len = fft_len` 有存起來，不然 class 外看不到。

---

### 3. `binaural_attention_model.py`

* Flatten/Unflatten 的通道數跟上面同理，要用動態計算或 `self.embed_dim//2`。

---

### 4. `MultiAttnBlock`

* `dtype=torch.complex64` 這行會報錯——`nn.Linear` 不支援複數 dtype。
  原本作者是把 real/imag 疊一起當 real tensor；保留舊寫法即可。

---

### 5. `configs/fft256.yaml`

* 其他 OK，但 `batch_size: 8` 若 VRAM 不夠可以降；
  `learning_rate_decay_steps` 應該是 epoch 編號，不是 list of list，確定 parser 支援。

---

### 6. smoke test

跑之前先加一段 shape print：

```python
with torch.no_grad():
    dummy = torch.randn(2, 2, 16000)  # B,Ch,T
    out = model(dummy)
    print(out.shape)
```

如果能過，才進 dataloader。

---
