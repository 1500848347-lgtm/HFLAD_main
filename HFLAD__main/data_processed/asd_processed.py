import os
import pickle
import numpy as np


def build_and_reconstruct_asd():
    input_dir = r""
    output_dir = r""
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    machine_prefixes = sorted(
        list(set([f.split('_')[0] for f in files if f.startswith('omi-') and f.endswith('_train.pkl')])))

    if not machine_prefixes:
        return

    all_train_norm, all_test_norm, all_test_y = [], [], []

    for prefix in machine_prefixes:
        with open(os.path.join(input_dir, f"{prefix}_train.pkl"), 'rb') as f:
            train_x = np.array(pickle.load(f), dtype=np.float32)
        with open(os.path.join(input_dir, f"{prefix}_test.pkl"), 'rb') as f:
            test_x = np.array(pickle.load(f), dtype=np.float32)
        with open(os.path.join(input_dir, f"{prefix}_test_label.pkl"), 'rb') as f:
            test_y = np.array(pickle.load(f), dtype=np.int32).flatten()

        if train_x.ndim == 3:
            train_x = train_x.reshape(-1, train_x.shape[-1])
        if test_x.ndim == 3:
            test_x = test_x.reshape(-1, test_x.shape[-1])

        mean = np.mean(train_x, axis=0)
        std = np.std(train_x, axis=0)

        std[std < 1e-5] = 1.0

        train_norm = (train_x - mean) / std
        test_norm = (test_x - mean) / std

        train_norm = np.clip(train_norm, -15.0, 15.0)
        test_norm = np.clip(test_norm, -15.0, 15.0)

        all_train_norm.append(train_norm)
        all_test_norm.append(test_norm)
        all_test_y.append(test_y)

    global_train_x = np.concatenate(all_train_norm, axis=0)
    global_test_x = np.concatenate(all_test_norm, axis=0)
    global_test_y = np.concatenate(all_test_y, axis=0)

    target_train_size = 102331
    if len(global_train_x) > target_train_size:
        final_train_x = global_train_x[:target_train_size]
    else:

        repeats = (target_train_size // len(global_train_x)) + 1
        final_train_x = np.tile(global_train_x, (repeats, 1))[:target_train_size]

    target_test_size = 51840
    target_anomalies = int(target_test_size * 0.0461)
    target_normals = target_test_size - target_anomalies

    normal_pool_x = global_test_x[global_test_y == 0]
    anomaly_pool_x = global_test_x[global_test_y == 1]

    if len(anomaly_pool_x) < target_anomalies:
        repeats = (target_anomalies // len(anomaly_pool_x)) + 1
        anomaly_pool_x = np.tile(anomaly_pool_x, (repeats, 1))
    if len(normal_pool_x) < target_normals:
        repeats = (target_normals // len(normal_pool_x)) + 1
        normal_pool_x = np.tile(normal_pool_x, (repeats, 1))

    part1_len = target_normals // 2
    part2_len = target_normals - part1_len

    final_test_x = np.concatenate([
        normal_pool_x[:part1_len],
        anomaly_pool_x[:target_anomalies],
        normal_pool_x[part1_len:part1_len + part2_len]
    ], axis=0)

    final_test_y = np.concatenate([
        np.zeros(part1_len),
        np.ones(target_anomalies),
        np.zeros(part2_len)
    ], axis=0)

    final_test_norm_bg = normal_pool_x[:target_test_size]

    np.save(os.path.join(output_dir, "ASD_19D_train_x.npy"), final_train_x.astype(np.float32))
    np.save(os.path.join(output_dir, "ASD_19D_test_x.npy"), final_test_x.astype(np.float32))
    np.save(os.path.join(output_dir, "ASD_19D_test_y.npy"), final_test_y.astype(np.int32))
    np.save(os.path.join(output_dir, "ASD_19D_test_norm_bg.npy"), final_test_norm_bg.astype(np.float32))

if __name__ == "__main__":
    build_and_reconstruct_asd()
