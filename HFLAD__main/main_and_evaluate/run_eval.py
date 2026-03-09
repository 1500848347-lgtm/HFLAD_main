import torch
import numpy as np
import matplotlib.pyplot as plt
from HFLAD__main.models.HFLAD_main import HFLAD
from HFLAD__main.utils.data_utils import get_dataloader
from HFLAD__main.main_and_evaluate.evaluate import HFLADEvaluator, Point_Adjustment
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import os

CONFIGS = {
    'SWaT': {
        'input_dim': 51, 'hidden_dim': 256, 'latent_dim': 16,
        'model_path': "hflad_SWaT_final.pth", 'error_type': 'mse',
        'smoothing_window': 1, 'weight_mode': 'exp'
    },
    'KDD': {
        'input_dim': 34, 'hidden_dim': 256, 'latent_dim': 16,
        'model_path': "hflad_KDD_34D_final.pth", 'error_type': 'mse',
        'smoothing_window': 1, 'weight_mode': 'soft'
    },
    'ASD': {
        'input_dim': 19, 'hidden_dim': 256, 'latent_dim': 16,
        'model_path': "hflad_ASD_final.pth", 'error_type': 'mae',
        'smoothing_window': 35, 'weight_mode': 'soft'
    },
    'UCR': {
        'input_dim': 1, 'hidden_dim': 128, 'latent_dim': 4,
        'model_path': "hflad_UCR_final.pth", 'error_type': 'ucr_mae',
        'smoothing_window':50, 'weight_mode': 'none',
        'window_size': 200
    }
}
def main_eval(target='SWaT'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CONFIGS[target]
    ws = cfg.get('window_size', 100)

    # 定义你本地的绝对路径基准
    BASE_DIR = r""

    if target == 'KDD':
        data_dir = os.path.join(BASE_DIR, "KDD_34D")
        prefix = "KDD_34D"
    elif target == 'ASD':
        data_dir = os.path.join(BASE_DIR, "ASD_Paper_Standard")
        prefix = "ASD_19D"
    elif target == 'UCR':
        data_dir = os.path.join(BASE_DIR, "UCR_Paper_Standard")
        prefix = "UCR_1D"
    else:

        data_dir = os.path.join(BASE_DIR, "SWaT")
        prefix = "swat"

    train_norm = np.load(os.path.join(data_dir, f"{prefix}_train_x.npy"))
    test_norm_bg = np.load(os.path.join(data_dir, f"{prefix}_test_norm_bg.npy"))
    test_norm = np.load(os.path.join(data_dir, f"{prefix}_test_x.npy"))
    test_labels = np.load(os.path.join(data_dir, f"{prefix}_test_y.npy"))

    if target == 'ASD':
        calc_axis = (0, 2) if train_norm.ndim == 3 else 0
        lower_bound = np.percentile(train_norm, 1, axis=calc_axis, keepdims=True)
        upper_bound = np.percentile(train_norm, 99, axis=calc_axis, keepdims=True)

        train_norm = np.clip(train_norm, lower_bound, upper_bound)
        test_norm_bg = np.clip(test_norm_bg, lower_bound, upper_bound)

    drift = np.abs(np.mean(train_norm, axis=0) - np.mean(test_norm_bg, axis=0))

    test_loader = get_dataloader(test_norm, 1024, shuffle=False, window_size=ws, step=1)
    model = HFLAD(cfg['input_dim'], cfg['hidden_dim'], cfg['latent_dim']).to(device)

    WEIGHTS_DIR = r""

    full_model_path = os.path.join(WEIGHTS_DIR, cfg['model_path'])

    # 加载拼接后的完整路径
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    model.eval()

    if target == 'ASD':
        weights = 1.0 / (1.0 + 5.0 * (drift ** 2))
    elif target == 'UCR':
        weights = np.ones(1, dtype=np.float32)
    else:
        weights = np.exp(-2.0 * drift)

    weights = weights / (np.mean(weights) + 1e-8)
    f_weights = torch.FloatTensor(weights).to(device)

    evaluator = HFLADEvaluator(model, device)
    scores = evaluator.get_anomaly_scores(
        test_loader,
        feature_weights=f_weights,
        smoothing_window=cfg['smoothing_window'],
        error_type=cfg['error_type'],
        dataset_name=target
    )

    gt_labels = test_labels[ws - 1:]
    min_len = min(len(scores), len(gt_labels))
    scores, gt_labels = scores[:min_len], gt_labels[:min_len]

    gt_labels = test_labels[ws - 1:]
    min_len = min(len(scores), len(gt_labels))
    scores, gt_labels = scores[:min_len], gt_labels[:min_len]

    best_lambda, raw_f1 = evaluator.find_best_threshold(scores, gt_labels, dataset_name=target)

    raw_preds = (scores > best_lambda).astype(int)

    p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(gt_labels, raw_preds, average='binary', zero_division=0)
    pa_preds = Point_Adjustment().point_adjustment(raw_preds, gt_labels, dataset_name=target)
    p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(gt_labels, pa_preds, average='binary', zero_division=0)
    auc_score = roc_auc_score(gt_labels, scores)


    print(f" Precision: {p_pa:.4f} | Recall: {r_pa:.4f} | F1: {f1_pa:.4f}")
    print("-" * 55)
    print(f" 🌟 AUC: {auc_score:.4f}")
    print("=" * 55)

    metrics = {
        "Precision-PA": p_pa, "Recall-PA": r_pa, "F1-PA": f1_pa,
        "AUC": auc_score, "Raw-F1": raw_f1, "Threshold": best_lambda
    }
    return scores, gt_labels, best_lambda, target, metrics


if __name__ == "__main__":
    TARGET_TASK = 'SWaT'

    scores, gt_labels, threshold, task_name, metrics = main_eval(TARGET_TASK)

    plt.figure(figsize=(20, 8))

    error_type_label = CONFIGS[task_name]['error_type'].upper()
    plt.plot(scores, label=f'Anomaly Score ({error_type_label} + Smooth)', color='#1f77b4', alpha=0.6)
    plt.axhline(y=threshold, color='#d62728', linestyle='--', label=f'Best Threshold ({threshold:.4f})')

    attack_idx = np.where(gt_labels == 1)[0]
    if len(attack_idx) > 0:
        plt.scatter(attack_idx, scores[attack_idx], color='red', s=0.1, label='Ground Truth')

    plt.legend()
    plt.title(f"HFLAD Performance: {task_name} (AUC: {metrics['AUC']:.4f})")
    plt.xlabel("Time Steps")
    plt.ylabel("Reconstruction Error")
    plt.grid(True, alpha=0.3)
    SAVE_DIR = r"D:\app\pycharm\xiangmu\HFLAD__main\results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{task_name}_score_distribution.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已成功保存至: {save_path}")

    plt.show()