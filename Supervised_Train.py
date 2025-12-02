from model import CNNModel
import torch
from torch import nn, optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

THRESHOLD = 5e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = 1
VERBOSE_BATCH = 100
SAVE_FREQ_EPOCH = 2
SAVE_PATH = "check_points/"

SPLIT_DIR = f"Pre_trained_Data/splitted_npy"
OBS_PATH = os.path.join(SPLIT_DIR, "observation.npy")
MASK_PATH = os.path.join(SPLIT_DIR, "action_mask.npy")
LABEL_PATH = os.path.join(SPLIT_DIR, "action.npy")

BATCH_SIZE = 128
MAX_EPOCHS = 1000
SUBSAMPLE = None
VAL_RATIO = 0.2  # 验证集比例，可按需调整

os.makedirs(SAVE_PATH, exist_ok=True)

for p in (OBS_PATH, MASK_PATH, LABEL_PATH):
    if not os.path.exists(p):
        raise RuntimeError(
    f"{p} not found. Run Pre_Data_Process.py to extract .npy from .npz")

print("Loading .npy with mmap...")
obs_arr = np.load(OBS_PATH, mmap_mode='r')
mask_arr = np.load(MASK_PATH, mmap_mode='r')
label_arr = np.load(LABEL_PATH, mmap_mode='r')
print("Loaded memmap shapes:", obs_arr.shape, mask_arr.shape, label_arr.shape)

class NpyMemmapDataset(Dataset):
    def __init__(self, obs_path, mask_path, label_path, indices):
        self.obs_path = obs_path
        self.mask_path = mask_path
        self.label_path = label_path
        self.indices = np.array(indices, dtype=np.int64)
        
        # These will be loaded in each worker process
        self.obs = None
        self.mask = None
        self.labels = None

    def __len__(self):
        return len(self.indices)

    def _load_data(self):
        # Load data if not already loaded (per-worker)
        if self.obs is None:
            self.obs = np.load(self.obs_path, mmap_mode='r')
            self.mask = np.load(self.mask_path, mmap_mode='r')
            self.labels = np.load(self.label_path, mmap_mode='r')

    def __getitem__(self, idx):
        self._load_data()
        i = int(self.indices[idx])
        
        obs = torch.from_numpy(self.obs[i]).to(torch.float32).clone()
        mask = torch.from_numpy(self.mask[i]).to(torch.float32).clone()
        label = torch.tensor(int(self.labels[i]), dtype=torch.int64)
        return obs, mask, label

num_samples = int(obs_arr.shape[0])
rng = np.random.default_rng()
all_indices = rng.permutation(num_samples)
if SUBSAMPLE:
    all_indices = all_indices[:min(SUBSAMPLE, len(all_indices))]
cut = int(len(all_indices) * (1 - VAL_RATIO))
train_idx_full = all_indices[:cut]
val_idx_full = all_indices[cut:]
print(f"Total samples: {len(all_indices)}; Train: {len(train_idx_full)}; Val: {len(val_idx_full)}")

if __name__ == '__main__':
    # 启用 cuDNN 最佳算法搜索以提升卷积性能
    torch.backends.cudnn.benchmark = True
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 提升并行的数据加载：Windows 使用 spawn，Dataset 仅持有路径，安全并行
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    num_workers = 2
    pin_memory = True if DEVICE == "cuda" else False
    print("\n=== Single Train/Val Split | train", len(train_idx_full), "val", len(val_idx_full), "===")
    train_ds = NpyMemmapDataset(OBS_PATH, MASK_PATH, LABEL_PATH, train_idx_full)
    val_ds = NpyMemmapDataset(OBS_PATH, MASK_PATH, LABEL_PATH, val_idx_full)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = CNNModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=8e-3)
    # 简单 warmup + cosine 调度：前 5 个 epoch 线性升至目标 lr，然后余弦衰减
    WARMUP_EPOCHS = 5
    TOTAL_EPOCHS = MAX_EPOCHS
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        # 余弦：从 1 -> 0.1 保留尾部学习率
        progress = (epoch - WARMUP_EPOCHS) / max(1, (TOTAL_EPOCHS - WARMUP_EPOCHS))
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return 0.1 + 0.9 * cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_val_loss = float("inf")
    save_mark = 0
    print("Load Succeed. Starting training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        save_mark += 1
        model.train()
        running = 0.0
        total_train = 0
        for batch_idx, (obs_b, mask_b, y_b) in enumerate(tqdm(train_loader, total=len(train_loader),
                                       desc=f"Train | epoch {epoch}")):
            obs_b = obs_b.to(DEVICE, non_blocking=True)
            mask_b = mask_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            if DEVICE == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits, _ = model({"observation": obs_b,
                                       "action_mask": mask_b})
                    loss = loss_fn(logits, y_b)
            else:
                logits, _ = model({"observation": obs_b,
                                   "action_mask": mask_b})
                loss = loss_fn(logits, y_b)
            loss.backward()
            optimizer.step()

            bs = obs_b.shape[0]
            running += loss.item() * bs
            total_train += bs
            del obs_b, mask_b, y_b, logits, loss

            if VERBOSE_BATCH and (batch_idx + 1) % VERBOSE_BATCH == 0:
                print(f"Train batch {batch_idx+1}/{len(train_loader)} | epoch {epoch} | ",
                      f"avg_loss={running/max(1,total_train):.6f}", sep='')

        train_loss = running / max(1, total_train)

        model.eval()
        val_running = 0.0
        total_val = 0
        with torch.inference_mode():
            for batch_idx, (obs_b, mask_b, y_b) in enumerate(tqdm(val_loader, total=len(val_loader),
                                           desc=f"Eval  | epoch {epoch}")):
                obs_b = obs_b.to(DEVICE, non_blocking=True)
                mask_b = mask_b.to(DEVICE, non_blocking=True)
                y_b = y_b.to(DEVICE, non_blocking=True)
                if DEVICE == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        logits, _ = model({"observation": obs_b,
                                           "action_mask": mask_b})
                        loss = loss_fn(logits, y_b)
                else:
                    logits, _ = model({"observation": obs_b,
                                       "action_mask": mask_b})
                    loss = loss_fn(logits, y_b)
                bs = obs_b.shape[0]
                val_running += loss.item() * bs
                total_val += bs
                del obs_b, mask_b, y_b, logits, loss

                if VERBOSE_BATCH and (batch_idx + 1) % VERBOSE_BATCH == 0:
                    print(f"Eval  batch {batch_idx+1}/{len(val_loader)} | epoch {epoch} | ",
                          f"avg_loss={val_running/max(1,total_val):.6f}", sep='')
        val_loss = val_running / max(1, total_val)

        if epoch % VERBOSE == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss and save_mark >= SAVE_FREQ_EPOCH:
            best_val_loss = val_loss
            save_mark = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_PATH, f"spv_model_epoch{epoch}_{val_loss:.4f}.pt"))

        # 学习率调度步进（在早停判断前）
        scheduler.step()
        if train_loss < THRESHOLD or val_loss < THRESHOLD:
            print(f"Early stopping epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            break

    print(f"Training finished. Best val loss: {best_val_loss:.6f}")
# ...existing code...