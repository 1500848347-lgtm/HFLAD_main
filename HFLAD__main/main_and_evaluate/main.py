import numpy as np
import os
import torch
from HFLAD__main.main_and_evaluate.run_experiment import run_full_hflad_pipeline

if __name__ == "__main__":
    configs = {
        'SWaT': {
            'dataset_name': 'SWaT', 'input_dim': 51, 'epochs': 30, 'error_type': 'mse',
            'mask_count': 5, 'smoothing_window': 5, 'step': 2, 'hidden_dim': 256,
            'latent_dim': 16, 'window_size': 100, 'batch_size': 1024
        },
        'MSL': {
            'dataset_name': 'MSL', 'input_dim': 55, 'epochs': 50, 'error_type': 'mae',
            'mask_count': 0, 'smoothing_window': 50, 'step': 1, 'hidden_dim': 256,
            'latent_dim': 16, 'window_size': 100, 'batch_size': 1024
        },
        'KDD': {
            'dataset_name': 'KDD_34D', 'input_dim': 34, 'epochs': 30, 'error_type': 'mse',
            'mask_count': 0, 'smoothing_window': 3, 'step': 1, 'hidden_dim': 256,
            'latent_dim': 16, 'window_size': 100, 'batch_size': 1024
        },
        'ASD': {
            'dataset_name': 'ASD', 'input_dim': 19, 'epochs': 40, 'error_type': 'mae',
            'mask_count': 0, 'smoothing_window': 15, 'step': 1, 'hidden_dim': 256,
            'latent_dim': 16, 'window_size': 100, 'batch_size': 1024
        },
        'UCR': {
            'dataset_name': 'UCR',
            'input_dim': 1,
            'epochs': 35,
            'error_type': 'ucr_mae',
            'mask_count': 0,
            'smoothing_window':5,
            'step': 1, 'hidden_dim': 128, 'latent_dim': 4, 'window_size': 200, 'batch_size': 1024
        }
    }
    target_dataset = 'UCR'
    current_cfg = configs[target_dataset]

    if target_dataset == 'KDD':
        data_dir = f"../../HF_qinli/KDD_34D"
        prefix = "KDD_34D"
    elif target_dataset == 'ASD':
        data_dir = f"../data_processed/ASD_Paper_Standard"
        prefix = "ASD_19D"
    elif target_dataset == 'UCR':
        data_dir = f"../data_processed/UCR_Paper_Standard"
        prefix = "UCR_1D"
    else:
        data_dir = f"data_processed/{target_dataset}"
        prefix = target_dataset

    train_norm = np.load(os.path.join(data_dir, f"{prefix}_train_x.npy"))
    test_norm = np.load(os.path.join(data_dir, f"{prefix}_test_x.npy"))
    test_labels = np.load(os.path.join(data_dir, f"{prefix}_test_y.npy"))
    test_norm_bg = np.load(os.path.join(data_dir, f"{prefix}_test_norm_bg.npy"))

    if target_dataset == 'ASD':
        calc_axis = (0, 2) if train_norm.ndim == 3 else 0
        lower_bound = np.percentile(train_norm, 1, axis=calc_axis, keepdims=True)
        upper_bound = np.percentile(train_norm, 99, axis=calc_axis, keepdims=True)
        train_norm = np.clip(train_norm, lower_bound, upper_bound)
        test_norm_bg = np.clip(test_norm_bg, lower_bound, upper_bound)
        test_norm = np.clip(test_norm, lower_bound, upper_bound)

    n_features = current_cfg['input_dim']
    weights = np.ones(n_features, dtype=np.float32)

    if target_dataset == 'SWaT':
        axis_to_mean = (0, 2) if train_norm.ndim == 3 else 0
        drift = np.abs(np.mean(train_norm, axis=axis_to_mean) - np.mean(test_norm_bg, axis=axis_to_mean))
        base_weights = 1.0 / (1.0 + 5.0 * (drift ** 2))
        if current_cfg['mask_count'] > 0:
            top_drift_indices = np.argsort(drift)[-current_cfg['mask_count']:]
            for idx in top_drift_indices:
                base_weights[idx] = 0.0
        weights = base_weights / (np.mean(base_weights) + 1e-8)

    elif target_dataset in ['MSL', 'SMAP']:
        discrete_count = 0
        for i in range(n_features):
            feature_data = train_norm[:, i, :] if train_norm.ndim == 3 else train_norm[:, i]
            unique_vals = len(np.unique(feature_data))
            if unique_vals < 15:
                weights[i] = 0.0
                discrete_count += 1
        weights = weights / (np.mean(weights) + 1e-8)

    elif target_dataset == 'KDD':
        super_low_count = heavy_down_count = 0
        for i in range(n_features):
            feature_data = train_norm[:, i, :] if train_norm.ndim == 3 else train_norm[:, i]
            std_val = np.std(feature_data)
            unique_vals = len(np.unique(feature_data))
            if std_val < 1e-4:
                weights[i] = 0.001
                super_low_count += 1
            elif unique_vals < 15:
                weights[i] = 0.02
                heavy_down_count += 1
        weights = weights / (np.mean(weights) + 1e-8)

    elif target_dataset == 'ASD':
        axis_to_mean = (0, 2) if train_norm.ndim == 3 else 0
        drift = np.abs(np.mean(train_norm, axis=axis_to_mean) - np.mean(test_norm_bg, axis=axis_to_mean))
        weights = 1.0 / (1.0 + 10.0 * (drift ** 2))

        weights = weights / (np.mean(weights) + 1e-8)

    elif target_dataset == 'UCR':
        weights = np.ones(1, dtype=np.float32)

    feature_weights = torch.FloatTensor(weights)
    current_cfg['feature_weights'] = feature_weights
    model = run_full_hflad_pipeline(train_norm, test_norm_bg, test_norm, test_labels, current_cfg)