import torch
import torch.nn as nn


class SRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.gate_linear = nn.Linear(input_dim + latent_dim + hidden_dim, 2 * hidden_dim)
        self.temp_state_linear = nn.Linear(input_dim + latent_dim + hidden_dim, hidden_dim)
        self.z_params_net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # 输出 mu 和 log_var
        )
    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std
    def forward(self, y_t, z_prev, d_prev):
        combined_gate = torch.cat([y_t, z_prev, d_prev], dim=-1)
        gates = torch.sigmoid(self.gate_linear(combined_gate))
        u_t, r_t = gates.chunk(2, dim=-1)
        combined_temp = torch.cat([y_t, z_prev, r_t * d_prev], dim=-1)
        d_prime_t = torch.tanh(self.temp_state_linear(combined_temp))
        d_t = u_t * d_prev + (1 - u_t) * d_prime_t
        combined_z = torch.cat([z_prev, d_t], dim=-1)
        z_params = self.z_params_net(combined_z)
        mu, log_var = z_params.chunk(2, dim=-1)
        z_t = self.reparameterize(mu, log_var)
        return z_t, d_t, mu, log_var

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(FeatureEncoder, self).__init__()
        self.cell = SRNNCell(input_dim, hidden_dim, latent_dim)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, y_seq):
        batch_size, seq_len, _ = y_seq.size()
        device = y_seq.device
        z_t = torch.zeros(batch_size, self.latent_dim).to(device)
        d_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        z_list, mu_list, log_var_list = [], [], []
        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            z_t, d_t, mu, log_var = self.cell(y_t, z_t, d_t)

            z_list.append(z_t)
            mu_list.append(mu)
            log_var_list.append(log_var)

        return torch.stack(z_list, dim=1), torch.stack(mu_list, dim=1), torch.stack(log_var_list, dim=1)