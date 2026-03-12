import numpy as np
import pickle
import os

def load_wesad_subject(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

def preprocess_wesad_for_hflad_lite():
    # 使用你电脑上正确的绝对路径
    wesad_root = r""
    out_dir = r""
    os.makedirs(out_dir, exist_ok=True)

    train_data_list = []
    test_data_list = []
    test_label_list = []

    train_subjects = ['S2', 'S3', 'S4', 'S5', 'S6']
    test_subjects = ['S16', 'S17']

    for subj in train_subjects:
        file_path = os.path.join(wesad_root, subj, f"{subj}.pkl")
        if not os.path.exists(file_path): continue
        data = load_wesad_subject(file_path)
        chest_data = np.concatenate([
            data['signal']['chest']['ACC'], data['signal']['chest']['ECG'],
            data['signal']['chest']['EDA'], data['signal']['chest']['EMG'],
            data['signal']['chest']['Resp'], data['signal']['chest']['Temp']
        ], axis=1)
        labels = data['label']

        baseline_mask = (labels == 1)
        train_data_list.append(chest_data[baseline_mask])

    for subj in test_subjects:
        file_path = os.path.join(wesad_root, subj, f"{subj}.pkl")
        if not os.path.exists(file_path): continue
        data = load_wesad_subject(file_path)
        chest_data = np.concatenate([
            data['signal']['chest']['ACC'], data['signal']['chest']['ECG'],
            data['signal']['chest']['EDA'], data['signal']['chest']['EMG'],
            data['signal']['chest']['Resp'], data['signal']['chest']['Temp']
        ], axis=1)
        labels = data['label']

        # 引入 Amusement(3) 和 Meditation(4) 作为正常数据，稀释异常比例
        test_mask = (labels == 1) | (labels == 2) | (labels == 3) | (labels == 4)
        valid_chest_data = chest_data[test_mask]
        valid_labels = labels[test_mask]

        binary_labels = np.where(valid_labels == 2, 1, 0)
        test_data_list.append(valid_chest_data)
        test_label_list.append(binary_labels)

    final_train_data = np.concatenate(train_data_list, axis=0)
    final_test_data = np.concatenate(test_data_list, axis=0)
    final_test_labels = np.concatenate(test_label_list, axis=0)

    DOWN_RATE = 70
    final_train_data = final_train_data[::DOWN_RATE]
    final_test_data = final_test_data[::DOWN_RATE]
    final_test_labels = final_test_labels[::DOWN_RATE]
    mean = np.mean(final_train_data, axis=0)
    std = np.std(final_train_data, axis=0)
    final_train_data = (final_train_data - mean) / (std + 1e-8)
    final_test_data = (final_test_data - mean) / (std + 1e-8)

    prefix = "WESAD"
    np.save(os.path.join(out_dir, f"{prefix}_train.npy"), final_train_data)
    np.save(os.path.join(out_dir, f"{prefix}_test.npy"), final_test_data)
    np.save(os.path.join(out_dir, f"{prefix}_test_label.npy"), final_test_labels)

if __name__ == "__main__":
    preprocess_wesad_for_hflad_lite()