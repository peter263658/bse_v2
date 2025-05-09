# TODO 
1. SegSNR issue
2. change vctk test dataset is spilt to snr
3. check if the model setup is the same as paper
4. run eval in fewer num samples



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




## 以下分析分兩部份：

## 1 重新訓練時必備（或仍缺漏）的檔案／指令

| 類別               | 原作者 BCCTN repo（可直接參考）([GitHub][1])                                      | 你目前 bse\_v2 repo 情況 ([GitHub][2])                                           | 若要「能完整 retrain」需再補齊或修正                                                                                                                               |
| ---------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **資料準備**         | `DCNN/datasets/` + `prepare_data*.py` 產生 VCTK/NOISEX-92 ➜ HRIR 合成之雙耳訓練集 | 已有 `prepare_data*.py`，但 **缺 `dataset/speech_dataset.yaml` 與 HRIR 路徑設定**     | 1. 建立 `config/dataset/speech_dataset.yaml`，把 *clean/noisy* train/val/test 的實際資料夾寫進去。<br>2. 確保 HRIR（CIPIC ID 28）wav 檔路徑正確。                            |
| **環境/相依套件**      | `requirements.txt`（PyTorch 1.10、pytorch-lightning 1.9、torchaudio 0.10…） | 你的 `environment.yml` 已加 GPU-12.4 版；**仍漏 `lightning<=2.0` 與 `librosa 0.10`** | 以 **mamba**：<br>`mamba install pytorch-lightning=1.9 librosa=0.10 --channel conda-forge`                                                             |
| **訓練腳本**         | `train.py` 直接吃 Hydra `config/`                                          | 你已有 `run_train.sh`，但指向的 `config/config.yaml` 把 accelerator 設 *cpu*          | - 把 `training.accelerator: gpu` 與 `strategy: ddp`（或 `ddp_spawn`）改成符合你的 H100 叢集。<br>- 更新 `batch_size`（原論文用 8，H100 可提到 32）。                            |
| **檢查點存檔**        | `DCNN/Checkpoints/Trained_model.ckpt` 範例                                | 有同名檔，但若要從頭訓練可刪除或指定 `train_checkpoint_path: null`                            | 無                                                                                                                                                    |
| **評估/Benchmark** | `mod_eval.py` + `run_eval.sh` 產生 SegSNR / MBSTOI / ILD/IPD              | 你有 `eval.py`，功能齊，但 **needs STFT 參數一致檢查腳本**                                  | （a）把 `verify_dataset/verify_dataset.py` 加入 CI；（b）在 `eval.py` 同步 `n_fft=512, win=400, hop=100`；（c）更新 `run_eval.sh` 把 `--checkpoint` 換成你 retrain 後的檔名。 |

---

## 2 「預訓模型設定」與論文差異，須調整的參數

| 參數                                   | 論文設定 (ICASSP 2024) ([GitHub][3]) | 你 repo 目前預設                                                        | 修正建議                                                                            |
| ------------------------------------ | -------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **Multi-Head Attention：`num_heads`** | 32 heads                         | `model.yaml`（舊版）記為 **8**                                           | 改回 32，並同步調整與 `embed_dim` 的可整除性。                                                 |
| **Attention embedding dim**          | 512（實部）+ 512（虛部）                 | 256 (+256)（從 `binaural_attention_model.py` 中 `x_attn[:,:128]` 可推斷） | `embed_dim: 512`；`linear_out_features: 1024`                                    |
| **Attention hidden size**            | 128                              | 64                                                                 | `hidden_size: 128`                                                              |
| **Encoder/Decoder channel 深度**       | `[16, 32, 64, 128, 256, 256]`    | 少一層、最大 128                                                         | 依論文把第 6 層 filters 提到 256；stride 保持 *(2,f) × (1,t)*                              |
| **Loss 權重**                          | `{α,β,γ,κ} = {1, 10, 1, 10}`     | 你的 `config/config.yaml` 留 default（全 1）                             | 把 `snr_loss_weight: 1`、`stoi_weight: 10`、`ild_weight: 1`、`ipd_weight: 10` 寫進設定。 |
| **FFT/window/hop**                   | 512 / 25 ms / 6.25 ms            | 你的 `prepare_data_fixed.py` 用 512/512/128 → **win\_len 錯**          | `win_length: 400`、`hop_length: 100`（16 kHz 時）。                                  |
| **Early-stopping**                   | patience = 3 epoch               | 你關閉                                                                | 建議開啟以免 over-fit。                                                                |

> ✅ **修改位置**：`config/model.yaml`, `config/training.yaml`, `DCNN/models/…`
> ✅ **re-export 既有 checkpoint 無效**―調參後請從頭訓練或加 `ckpt_path=None`。

---

### 快速檢查流程

1. **資料完整性**

   ```bash
   ./verify_dataset/verify_dataset.py --config config/dataset/speech_dataset.yaml
   ```
2. **環境就緒 (mamba)**

   ```bash
   mamba env create -f environment.yml
   mamba activate bcctn
   ```
3. **開始訓練**

   ```bash
   ./run_train.sh  # 內部已指定 GPU & 正確 config
   ```
4. **驗證與生成報告**

   ```bash
   ./run_eval.sh  # 會輸出 MBSTOI / ΔSegSNR / LILD / LIPD
   ./benchmark_bcctn.py --checkpoint path/to/last.ckpt
   ```

這樣即可確保你的重新訓練流程與論文設定一致，且後續評估指標能與 Table 1/2 對齊。若還有細部疑問（如 HRIR 下載或 STFT CUDA 加速），再告訴我！

[1]: https://github.com/VikasTokala/BCCTN "GitHub - VikasTokala/BCCTN"
[2]: https://github.com/peter263658/bse_v2 "GitHub - peter263658/bse_v2"
[3]: https://github.com/peter263658/bse_v2/blob/main/DCNN/models/binaural_attention_model.py?raw=true "github.com"


above is the answer from chatgpt, your can just choose part of the suggestion based on my current provide files in project knowledge and I can provide more detailed:

1.  my current pre-trained model has different parameter setup with paper, but I don't want to train on scratch, if any method I can change the model architecture but still re-train from the pre-trained model? 
2. you also can refer the paper I provide to you.