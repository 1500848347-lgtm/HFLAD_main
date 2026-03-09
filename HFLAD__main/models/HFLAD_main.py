import torch
import torch.nn as nn
import torch.nn.functional as F
from HFLAD__main.models.hvae_gen import HVAEGenerator
from HFLAD__main.models.srnn_cell import FeatureEncoder
from HFLAD__main.models.tcn_module import HierarchicalTimeEncoder


class HFLAD(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(HFLAD, self).__init__()
        self.time_encoder = HierarchicalTimeEncoder(input_dim, hidden_dim)
        self.feature_encoder_a = FeatureEncoder(hidden_dim, hidden_dim, latent_dim)
        self.feature_encoder_b = FeatureEncoder(hidden_dim, hidden_dim, latent_dim)
        self.feature_encoder_c = FeatureEncoder(hidden_dim, hidden_dim, latent_dim)
        self.generator = HVAEGenerator(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        x_orig = x.transpose(1, 2)
        a, b, c = self.time_encoder(x)
        a, b, c = a.transpose(1, 2), b.transpose(1, 2), c.transpose(1, 2)
        z1, mu1, log_var1 = self.feature_encoder_a(a)
        z2, mu2, log_var2 = self.feature_encoder_b(b)
        z3, mu3, log_var3 = self.feature_encoder_c(c)
        x_hat = self.generator(z1, z2, z3)
        (mu_p1, log_var_p1), (mu_p2, log_var_p2) = self.generator.get_prior_params(z2, z3)

        return {
            "x_hat": x_hat, "x_orig": x_orig,
            "post_params": [(mu1, log_var1), (mu2, log_var2), (mu3, log_var3)],
            "prior_params": [(mu_p1, log_var_p1), (mu_p2, log_var_p2)]
        }

    def compute_loss(self, outputs, kl_weight=1.0, feature_weights=None, error_type='mse'):
        (mu1, lv1), (mu2, lv2), (mu3, lv3) = outputs["post_params"]
        (mu_p1, lvp1), (mu_p2, lvp2) = outputs["prior_params"]
        eps = 1e-8
        if error_type == 'mse':
            diff = (outputs["x_hat"] - outputs["x_orig"]) ** 2
        else:
            diff = torch.abs(outputs["x_hat"] - outputs["x_orig"])
        diff = torch.clamp(diff, max=100.0)
        if feature_weights is not None:
            num_features = feature_weights.shape[0]
            if diff.shape[1] == num_features:
                w = feature_weights.view(1, -1, 1)
            elif diff.shape[2] == num_features:
                w = feature_weights.view(1, 1, -1)
            else:
                w = feature_weights.view(1, -1, 1)

            diff = diff * w.to(diff.device)

        recon_loss = torch.mean(diff)

        total_elements = mu1.numel()

        def compute_kl(m, l, mp, lp):
            return 0.5 * torch.sum(lp - l + (l.exp() + (m - mp).pow(2)) / (lp.exp() + eps) - 1) / total_elements

        kl2 = compute_kl(mu2, lv2, mu_p2, lvp2)
        kl1 = compute_kl(mu1, lv1, mu_p1, lvp1)
        kl3 = -0.5 * torch.sum(1 + lv3 - mu3.pow(2) - lv3.exp()) / total_elements

        return recon_loss + kl_weight * (kl1 + kl2 + kl3)

    def compute_anomaly_score(self, x_orig, x_hat, error_type='mse'):
        if error_type == 'mse':
            score = torch.mean((x_orig - x_hat) ** 2, dim=[1, 2])
        elif error_type == 'mae':
            score = torch.mean(torch.abs(x_orig - x_hat), dim=[1, 2])
        elif error_type == 'ucr_mae':
            diff_base = torch.abs(x_orig - x_hat)

            grad_orig = x_orig[:, 1:, :] - x_orig[:, :-1, :]
            grad_hat = x_hat[:, 1:, :] - x_hat[:, :-1, :]
            shape_diff = torch.abs(grad_hat - grad_orig)
            shape_diff = F.pad(shape_diff, (0, 0, 1, 0))
            acc_orig = grad_orig[:, 1:, :] - grad_orig[:, :-1, :]
            acc_hat = grad_hat[:, 1:, :] - grad_hat[:, :-1, :]
            acc_diff = torch.abs(acc_hat - acc_orig)
            acc_diff = F.pad(acc_diff, (0, 0, 2, 0))  # 补齐 2 个时间步
            diff = diff_base + 5.0 * shape_diff + 20.0 * acc_diff
            score = torch.mean(diff, dim=[1, 2])
        else:
            score = torch.mean((x_orig - x_hat) ** 2, dim=[1, 2])

        return score
