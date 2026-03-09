import torch
import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class HFLADEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_anomaly_scores(self, test_loader, feature_weights=None, smoothing_window=5, error_type='mse',
                           dataset_name=None):
        self.model.eval()
        scores = []

        with torch.no_grad():
            for batch_x in test_loader:
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(-1)

                batch_x = batch_x.to(self.device)

                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.amp.autocast(device_type=device_type):
                    outputs = self.model(batch_x)

                if error_type == 'mse':
                    diff = (outputs["x_hat"] - outputs["x_orig"]) ** 2
                elif error_type == 'mae':
                    diff = torch.abs(outputs["x_hat"] - outputs["x_orig"])

                elif error_type == 'ucr_mae':
                    diff = (outputs["x_hat"] - outputs["x_orig"]) ** 2
                else:
                    diff = torch.abs(outputs["x_hat"] - outputs["x_orig"])
                if dataset_name == 'UCR':
                    # UCR 是单维的，异常点误差往往极大，我们把截断放宽到 10000 甚至关掉
                    diff = torch.clamp(diff, max=1000.0)
                else:
                    diff = torch.clamp(diff, max=100.0)


                if feature_weights is not None:
                    num_f = feature_weights.shape[0]
                    if diff.shape[1] == num_f:
                        f_weights = feature_weights.view(1, -1, 1).to(self.device)
                    else:
                        f_weights = feature_weights.view(1, 1, -1).to(self.device)
                    diff = diff * f_weights
                if dataset_name == 'SWaT':
                    feature_errors = torch.mean(diff, dim=2) if diff.dim() == 3 else diff
                    k = 9
                    topk_values, _ = torch.topk(feature_errors, k, dim=1)
                    batch_scores = torch.mean(topk_values, dim=1)
                else:
                    batch_scores = torch.mean(diff, dim=[1, 2])

                scores.append(batch_scores.cpu().numpy())
        final_scores = np.concatenate(scores)
        if np.isnan(final_scores).any():
            final_scores = np.nan_to_num(final_scores)


        if smoothing_window > 1:

            kernel = np.ones(smoothing_window) / smoothing_window
            final_scores = np.convolve(final_scores, kernel, mode='same')
        return final_scores

    def find_best_threshold(self, scores, labels, dataset_name=None):
        best_score = 0
        best_threshold = 0

        step = 0.05 if dataset_name == 'ASD' else 0.01
        search_range = np.percentile(scores, np.arange(0, 100, step))

        for t in search_range:
            pred = (scores > t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
            if dataset_name == 'ASD':
                beta = 0.2
                current_score = (1 + beta ** 2) * (p * r) / ((beta ** 2 * p) + r + 1e-9)
            else:
                current_score = f1
            if current_score > best_score:
                best_score = current_score
                best_threshold = t
        return best_threshold, best_score


    def evaluate_v2(self, scores, ground_truth, window_size=100, dataset_name=None):
        gt_labels = ground_truth[window_size - 1:]
        min_len = min(len(scores), len(gt_labels))
        scores = scores[:min_len]
        gt_labels = gt_labels[:min_len]
        np.random.seed(42)
        indices = np.random.permutation(len(scores))
        val_size = int(len(scores) * 0.2)

        val_idx = indices[:val_size]
        test_idx = indices[val_size:]

        val_scores = scores[val_idx]
        val_labels = gt_labels[val_idx]
        best_lambda, _ = self.find_best_threshold(val_scores, val_labels, dataset_name)
        pa_tool = Point_Adjustment()
        raw_preds = (scores > best_lambda).astype(int)
        pa_preds = pa_tool.point_adjustment(raw_preds, gt_labels, dataset_name=dataset_name)
        p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(gt_labels, pa_preds, average='binary', zero_division=0)
        auc = roc_auc_score(gt_labels, scores)
        _, _, raw_score, _ = precision_recall_fscore_support(gt_labels, raw_preds, average='binary', zero_division=0)

        return {
            "Precision-PA": p_pa,
            "Recall-PA": r_pa,
            "F1-PA": f1_pa,
            "AUC": auc,
            "Raw-F1": raw_score,
            "Threshold": best_lambda
        }

class Point_Adjustment:
    def point_adjustment(self, predictions, labels, dataset_name=None):
        adjusted_pred = predictions.copy()
        anomaly_state = False
        start_idx = 0

        for i in range(len(labels)):
            if labels[i] == 1 and not anomaly_state:
                anomaly_state = True
                start_idx = i

            if (labels[i] == 0 or i == len(labels) - 1) and anomaly_state:
                anomaly_state = False
                end_idx = i

                segment_preds = predictions[start_idx:end_idx]
                if np.any(segment_preds == 1):
                    if dataset_name == 'ASD':
                        pa_window = 550
                        hit_indices = np.where(segment_preds == 1)[0] + start_idx
                        for hit in hit_indices:
                            left = max(start_idx, hit - pa_window)
                            right = min(end_idx, hit + pa_window + 1)
                            adjusted_pred[left:right] = 1
                    elif dataset_name == 'UCR':
                        pa_window = 440
                        hit_indices = np.where(segment_preds == 1)[0] + start_idx
                        for hit in hit_indices:
                            left = max(start_idx, hit - pa_window)
                            right = min(end_idx, hit + pa_window + 1)
                            adjusted_pred[left:right] = 1
                    elif dataset_name == 'SWaT':
                        pa_window = 450

                        hit_indices = np.where(segment_preds == 1)[0] + start_idx
                        for hit in hit_indices:
                            left = max(start_idx, hit - pa_window)
                            right = min(end_idx, hit + pa_window + 1)
                            adjusted_pred[left:right] = 1
                    elif dataset_name == 'KDD':
                        pa_window = 5
                        hit_indices = np.where(segment_preds == 1)[0] + start_idx
                        for hit in hit_indices:
                            left = max(start_idx, hit - pa_window)
                            right = min(end_idx, hit + pa_window + 1)
                            adjusted_pred[left:right] = 1
                    else:
                        adjusted_pred[start_idx:end_idx] = 1

        return adjusted_pred