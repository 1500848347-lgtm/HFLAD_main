import torch
import numpy as np
import os
from HFLAD__main.train import train_hflad
from HFLAD__main.utils.data_utils import get_dataloader
from HFLAD__main.main_and_evaluate.evaluate import HFLADEvaluator, Point_Adjustment, EVAL_CONFIGS
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def run_full_hflad_pipeline(train_norm, test_norm_bg, test_norm, test_labels, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_ws = config['window_size']
    dataset = config['dataset_name']
    m_count = config.get('mask_count', 5)
    e_type = config.get('error_type', 'mse')
    s_window = config.get('smoothing_window', 5)

    if 'feature_weights' in config:
        f_weights = config['feature_weights'].to(device)
    else:
        drift = np.abs(np.mean(train_norm, axis=0) - np.mean(test_norm_bg, axis=0))
        weights = 1.0 / (1.0 + 10.0 * (drift ** 2))

        if m_count > 0:
            top_drift_indices = np.argsort(drift)[-m_count:]
            for idx in top_drift_indices:
                weights[idx] = 0.0
        weights = weights / np.mean(weights)
        f_weights = torch.FloatTensor(weights).to(device)

    train_loader = get_dataloader(train_norm, config['batch_size'], shuffle=True, window_size=current_ws,
                                  step=config['step'])
    test_loader = get_dataloader(test_norm, config['batch_size'], shuffle=False, window_size=current_ws, step=1)

    model = train_hflad(
        train_loader, train_norm, test_norm_bg,
        config['input_dim'], config['hidden_dim'],
        config['latent_dim'], config['epochs'],
        mask_count=m_count,
        error_type=e_type,
        custom_weights=f_weights
    )

    save_path = f"hflad_{dataset}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[!] 权重已落盘: {os.path.abspath(save_path)}")
    torch.cuda.empty_cache()

    try:
        evaluator = HFLADEvaluator(model, device)
        scores = evaluator.get_anomaly_scores(
            test_loader,
            feature_weights=f_weights,
            error_type=e_type,
            smoothing_window=s_window,
            dataset_name=dataset
        )
        gt_labels = test_labels[current_ws - 1:]
        min_len = min(len(scores), len(gt_labels))
        scores, gt_labels = scores[:min_len], gt_labels[:min_len]

        eval_cfg = EVAL_CONFIGS.get(dataset, EVAL_CONFIGS['DEFAULT'])


        best_lambda, raw_f1 = evaluator.find_best_threshold(
            scores, gt_labels,
            step=eval_cfg['step'],
            beta=eval_cfg['beta']
        )

        raw_preds = (scores > best_lambda).astype(int)
        p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(gt_labels, raw_preds, average='binary',
                                                                  zero_division=0)
        pa_preds = Point_Adjustment().point_adjustment(
            raw_preds, gt_labels,
            pa_window=eval_cfg['pa_window']
        )
        p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(gt_labels, pa_preds, average='binary', zero_division=0)
        auc = roc_auc_score(gt_labels, scores)

        print(f" Precision-PA:  {p_pa:.4f} | Recall-PA:  {r_pa:.4f} | F1-PA:  {f1_pa:.4f}")
        print("-" * 55)
        print(f"  全局分类能力 AUC: {auc:.4f}")
        print("=" * 55)

    except Exception as e:
        print(f"\n[!] 评估阶段出错: {e}")

    return model
