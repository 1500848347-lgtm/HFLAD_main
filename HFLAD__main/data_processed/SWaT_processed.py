import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # 必须改回 MinMaxScaler 以确保数值安全
import os

def prepare_swat_offline_safe(train_csv, test_csv, save_dir="data_processed"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tr_df = pd.read_csv(train_csv).fillna(0)
    te_df = pd.read_csv(test_csv).fillna(0)
    tr_df.columns = tr_df.columns.str.strip()
    te_df.columns = te_df.columns.str.strip()

    all_normal = tr_df.iloc[:, 1:52].values.astype(float)
    attack_raw = te_df.iloc[:, 1:52].values.astype(float)
    attack_labels = (te_df.iloc[:, -1].astype(str).str.strip() != 'Normal').astype(int)
    train_x = all_normal[:475200]

    needed_normal_test = 449919 - len(attack_raw)
    test_normal_bg = all_normal[475200: 475200 + needed_normal_test]

    test_x = np.vstack([test_normal_bg, attack_raw])
    test_y = np.concatenate([np.zeros(len(test_normal_bg), dtype=int), attack_labels.values])

    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    test_normal_processed = test_x[:len(test_normal_bg)]

    np.save(os.path.join(save_dir, "swat_train_x.npy"), train_x)
    np.save(os.path.join(save_dir, "swat_test_x.npy"), test_x)
    np.save(os.path.join(save_dir, "swat_test_y.npy"), test_y)
    np.save(os.path.join(save_dir, "swat_test_norm_bg.npy"), test_normal_processed)

if __name__ == "__main__":
    prepare_swat_offline_safe("data/normal.csv", "data/attack.csv")