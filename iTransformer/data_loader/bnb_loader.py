import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class BNBUSDTDataset(Dataset):
    def __init__(self, csv_path, seq_len=720, pred_len=96, split="train"):
        # Read your absolute-path CSV
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        # Use only the 'close' column (add others if you like)
        series = df['close'].values.astype(np.float32)

        # Normalize
        self.mean = series.mean()
        self.std  = series.std()
        normed    = (series - self.mean) / self.std

        # Split indices
        n         = len(normed)
        train_cut = int(n * 0.7)
        val_cut   = int(n * 0.85)

        if split == "train":
            data = normed[:train_cut]
        elif split == "val":
            data = normed[train_cut:val_cut]
        else:
            data = normed[val_cut:]

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.data     = data

    def __len__(self):
        return len(self.data) - (self.seq_len + self.pred_len) + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return (
            torch.from_numpy(x).unsqueeze(-1),
            torch.from_numpy(y).unsqueeze(-1)
        )