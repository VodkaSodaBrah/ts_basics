import os
import sys

# ---- add project paths so "fedformer", "utils", etc. import cleanly ----
project_root = os.path.abspath("..")
fed_root     = os.path.join(project_root, "fedformer")
utils_root   = os.path.join(fed_root, "utils")
layers_root  = os.path.join(fed_root, "layers")

for p in (utils_root, layers_root, fed_root, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)
# -----------------------------------------------------------------------

from fedformer.utils.masking import LocalMask   # (only needed if you call it elsewhere)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def load_csv(path):
    df = pd.read_csv(path)                      # raw CSV
    if "timestamp" in df.columns:
        df.sort_values("timestamp", inplace=True)
    return df

def build_features(df):
    vals   = df["close"].values.reshape(-1, 1)  # “close” price
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vals)
    return scaled.astype(np.float32), scaler

def create_windows(data, I=96, O=96):
    X, Y = [], []
    total = len(data)
    for start in range(total - I - O + 1):
        X.append(data[start : start + I])
        Y.append(data[start + I : start + I + O])
    return np.stack(X), np.stack(Y)

if __name__ == "__main__":
    df           = load_csv("../../data/bnbusdt_1m.csv")
    series, _    = build_features(df)
    X, Y         = create_windows(series, I=96, O=96)
    torch.save((torch.from_numpy(X), torch.from_numpy(Y)),
               "../../data/windows.pt")
    print("Saved", X.shape, Y.shape)