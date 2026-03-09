import torch
import torch.nn as nn


class HVAEGenerator(nn.Module):

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(HVAEGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.p_z2_given_z3 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # 输出 mu 和 log_var
        )

        self.p_z1_given_z23 = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 输出重构的 x_hat
        )

    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z1, z2, z3):
        combined_z = torch.cat([z1, z2, z3], dim=-1)
        x_hat = self.decoder(combined_z)
        return x_hat

    def get_prior_params(self, z2, z3):
        params_z2 = self.p_z2_given_z3(z3)
        mu_p2, log_var_p2 = params_z2.chunk(2, dim=-1)
        log_var_p2 = torch.clamp(log_var_p2, min=-10, max=10)
        combined_z23 = torch.cat([z2, z3], dim=-1)
        params_z1 = self.p_z1_given_z23(combined_z23)
        mu_p1, log_var_p1 = params_z1.chunk(2, dim=-1)
        log_var_p1 = torch.clamp(log_var_p1, min=-10, max=10)
        return (mu_p1, log_var_p1), (mu_p2, log_var_p2)