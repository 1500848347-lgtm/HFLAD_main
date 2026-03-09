

import numpy as np
import os


def recover_original_2d(data_3d):

    first_window = data_3d[0, :, :].T
    rest_steps = data_3d[1:, :, -1]
    return np.concatenate([first_window, rest_steps], axis=0)

def rebuild_kdd_paper_standard():
    input_dir = r""
    output_dir = r""
    os.makedirs(output_dir, exist_ok=True)

    train_x_3d = np.load(os.path.join(input_dir, "KDD_train_x.npy"))
    test_x_3d = np.load(os.path.join(input_dir, "KDD_test_x.npy"))
    test_y_raw = np.load(os.path.join(input_dir, "KDD_test_y.npy")).reshape(-1)

    train_x = recover_original_2d(train_x_3d)
    test_x = recover_original_2d(test_x_3d)

    test_y = np.concatenate([np.zeros(99), test_y_raw])

    variances = np.var(train_x, axis=0)
    drop_indices = np.argsort(variances)[:5]
    keep_indices = [i for i in range(train_x.shape[1]) if i not in drop_indices]

    train_x = train_x[:, keep_indices]
    test_x = test_x[:, keep_indices]

    target_test_size = 24602
    target_anomalies = int(target_test_size * 0.0569)
    target_normals = target_test_size - target_anomalies

    normal_pool_x = test_x[test_y == 0]
    anomaly_pool_x = test_x[test_y == 1]

    part1_len = target_normals // 2
    part2_len = target_normals - part1_len

    test_x_final = np.concatenate([
        normal_pool_x[:part1_len],
        anomaly_pool_x[:target_anomalies],
        normal_pool_x[part1_len:part1_len + part2_len]
    ], axis=0)

    test_y_final = np.concatenate([
        np.zeros(part1_len),
        np.ones(target_anomalies),
        np.zeros(part2_len)
    ], axis=0)

    np.save(os.path.join(output_dir, "KDD_34D_train_x.npy"), train_x.astype(np.float32))
    np.save(os.path.join(output_dir, "KDD_34D_test_x.npy"), test_x_final.astype(np.float32))
    np.save(os.path.join(output_dir, "KDD_34D_test_y.npy"), test_y_final)
    np.save(os.path.join(output_dir, "KDD_34D_test_norm_bg.npy"), normal_pool_x[:target_test_size].astype(np.float32))


if __name__ == "__main__":
    rebuild_kdd_paper_standard()
