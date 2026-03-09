import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class AdaptiveWindowDataset(Dataset):
    def __init__(self, data, window_size=100, step=5):
        self.window_size = window_size
        self.step = step

        self.is_pre_windowed = (data.ndim == 3)

        if self.is_pre_windowed:
            self.data = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
            self.len = self.data.shape[0]
        else:
            self.data = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
            self.indices = np.arange(0, len(data) - window_size + 1, step)
            self.len = len(self.indices)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.is_pre_windowed:
            return self.data[index]
        else:
            start = self.indices[index]
            window = self.data[start: start + self.window_size]
            return window.transpose(1, 0)

def align_and_clean_swat(train_csv, test_csv):

    # 加载并立即处理检查出的 694万个 NaN 问题
    tr_df = pd.read_csv(train_csv).fillna(0)
    te_df = pd.read_csv(test_csv).fillna(0)

    tr_df.columns = tr_df.columns.str.strip()
    te_df.columns = te_df.columns.str.strip()

    train_raw = tr_df.iloc[:, 1:52].values.astype(float)
    test_raw = te_df.iloc[:, 1:52].values.astype(float)
    test_labels_raw = (te_df.iloc[:, -1].astype(str).str.strip() != 'Normal').astype(int)
    train_aligned = train_raw[::3][:475200]
    if len(test_raw) < 449919:
        needed = 449919 - len(test_raw)
        normal_back = train_raw[-needed:]
        test_aligned = np.vstack([normal_back, test_raw])
        labels_aligned = np.concatenate([np.zeros(needed), test_labels_raw])
    else:
        test_aligned = test_raw[:449919]
        labels_aligned = test_labels_raw[:449919]
    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_aligned)
    test_norm = scaler.transform(test_aligned)
    return train_norm, test_norm, labels_aligned

def get_dataloader(data, batch_size=1024, shuffle=True, window_size=100, step=5):
    dataset = AdaptiveWindowDataset(data, window_size=window_size, step=step)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )