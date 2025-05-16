import hydra

from omegaconf import DictConfig

from DCNN.datasets import create_torch_dataloaders
# from DCNN.trainer import DCNNTrainer
# import warnings
# warnings.simplefilter('ignore')
# import torch
# from DCNN.trainer import DCNNLightningModule

# torch.cuda.empty_cache()
# max_split_size_mb = 512
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb

import warnings, torch
from DCNN.trainer import DCNNTrainer, DCNNLightningModule

warnings.simplefilter("ignore")
# optional：限制一次 allocation，不是必需可拿掉
torch.cuda.empty_cache()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra



@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 

    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    dataset_train, dataset_val, dataset_test = create_torch_dataloaders(config)
    
    trainer = DCNNTrainer(config)

    trainer.fit(dataset_train, val_dataloaders=dataset_val)
    trainer.test(dataset_test)

def smoke_test(model: torch.nn.Module):
    print("Running smoke test on model...")
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(2, 2, 16000, device=device)  # B,Ch,T
        try:
            out = model(dummy)
            print(f"STFT shape            : {model.stft(dummy[:,0:1]).shape}")
            print(f"Model forward pass successful! Output shape: {out.shape}")
            if out.shape == (2, 2, 16000):  # Same shape as input
                print("✅ Output shape matches expected shape")
            else:
                print(f"⚠️ Output shape mismatch: expected (2, 2, 16000), got {out.shape}")
        except Exception as e:
            print(f"❌ Model forward pass failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

# if __name__ == "__main__":
#     model = DCNNLightningModule(config)
#     smoke_test(model.model)      
#     train()

if __name__ == "__main__":
    # 1️⃣ 先手動載入一次 config 做 smoke-test
    initialize(config_path="config", version_base="1.1")
    cfg = compose(config_name="config")          # 可加 overrides=sys.argv[1:]
    lit = DCNNLightningModule(cfg)
    smoke_test(lit.model)

    # 2️⃣ 清掉 Hydra 全域狀態，讓 @hydra.main 能接手
    GlobalHydra.instance().clear()

    # 3️⃣ 正式進入訓練（Hydra 會重新 parse CLI 參數）
    train()   
