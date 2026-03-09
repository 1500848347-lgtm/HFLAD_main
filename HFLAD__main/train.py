import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.amp import autocast, GradScaler
import csv
import numpy as np
from HFLAD__main.models.HFLAD_main import HFLAD

def train_hflad(train_loader, train_x_raw, test_norm_bg, input_dim, hidden_dim, latent_dim,
                epochs=100, mask_count=5, error_type='mse', custom_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HFLAD(input_dim, hidden_dim, latent_dim).to(device)
    if custom_weights is not None:
        f_weights = custom_weights.to(device)
    else:
        drift = np.abs(np.mean(train_x_raw, axis=0) - np.mean(test_norm_bg, axis=0))
        weights = 1.0 / (1.0 + 5.0 * (drift ** 2))
        if mask_count > 0:
            top_drift_indices = np.argsort(drift)[-mask_count:]
            for idx in top_drift_indices:
                weights[idx] = 0.0
        weights = weights / np.mean(weights)
        f_weights = torch.FloatTensor(weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.5)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    f = open("train_log.csv", "w", newline="")
    log_writer = csv.writer(f)
    log_writer.writerow(["epoch", "loss", "lr", "kl_w"])
    model.train()
    for epoch in range(epochs):
        kl_w = min(1.0, epoch / (epochs * 0.2))
        if error_type == 'ucr_mae':
            kl_w = kl_w * 3.0
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)

        for batch_x in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type):

                if error_type == 'ucr_mae':
                    mask = torch.ones_like(batch_x)
                    B, C, T = batch_x.shape
                    block_size = 40
                    num_blocks = 2

                    for _ in range(num_blocks):
                        starts = torch.randint(0, T - block_size, (B,))
                        for i in range(B):
                            mask[i, :, starts[i]:starts[i] + block_size] = 0.0

                    noisy_batch_x = batch_x * mask
                    outputs = model(noisy_batch_x)
                    outputs["x_orig"] = batch_x.transpose(1, 2)
                else:
                    outputs = model(batch_x)
                loss = model.compute_loss(
                    outputs,
                    kl_weight=kl_w,
                    feature_weights=f_weights,
                    error_type=error_type  # <--- 关键参数接力
                )

            if torch.isnan(loss) or loss.item() > 1e10:
                pbar.set_postfix_str("Warning: Instability detected, skipping...")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        avg_loss = epoch_loss / len(train_loader)
        log_writer.writerow([epoch + 1, avg_loss, optimizer.param_groups[0]['lr'], kl_w])
        scheduler.step()
        print(f"[*] Epoch {epoch + 1} 完成 | 平均 Loss: {avg_loss:.4f}")

    f.close()
    return model