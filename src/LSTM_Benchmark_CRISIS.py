# ============================================================
# LSTM Benchmark -- CALENDAR-SPLIT VERSION
# Mirrors NSVM_Model.py's structure, config, loss, and metrics
# exactly. Only the split logic differs from before: calendar-
# based (via data_loader.calendar_split_indices), so COVID and
# the 2022 bear market are genuinely out-of-sample here too.
# ============================================================

import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_rv_series, calendar_split_indices

# ---- Config (identical to NSVM_Model.py) ----
SEED = 42
LOOKBACK = 20
HORIZON = 1
LSTM_HIDDEN = 32
BATCH_SIZE = 128
EPOCHS = 300
TRAIN_MC_PATHS = 24
VAL_MC_PATHS = 100
TEST_MC_PATHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LAMBDA_CRPS = 0.5
EARLY_STOPPING_PATIENCE = 30
MIN_DELTA = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Data (real ES, calendar split) ----
log_rv, rv, T, dates = load_rv_series()
train_end, val_end = calendar_split_indices(dates)
print(f"Train ends: {dates[train_end-1].date()}  |  "
      f"Val ends: {dates[val_end-1].date()}  |  "
      f"Test: {dates[val_end].date()} to {dates[-1].date()}")

mean_y = log_rv[:train_end].mean()
std_y = log_rv[:train_end].std()
y = (log_rv - mean_y) / std_y


def make_sequences(series, lookback, horizon):
    X, Y, indices = [], [], []
    end = len(series) - lookback - horizon + 1
    for i in range(end):
        X.append(series[i:i + lookback])
        Y.append(series[i + lookback + horizon - 1])
        indices.append(i + lookback + horizon - 1)
    return np.asarray(X), np.asarray(Y), np.asarray(indices)


X_all, Y_all, target_idx = make_sequences(y, LOOKBACK, HORIZON)
train_mask = target_idx < train_end
val_mask = (target_idx >= train_end) & (target_idx < val_end)
test_mask = target_idx >= val_end


def tensor(x):
    return torch.tensor(x, dtype=torch.float32)


X_train = tensor(X_all[train_mask]); Y_train = tensor(Y_all[train_mask]).unsqueeze(-1)
X_val = tensor(X_all[val_mask]); Y_val = tensor(Y_all[val_mask]).unsqueeze(-1)
X_test = tensor(X_all[test_mask]); Y_test = tensor(Y_all[test_mask]).unsqueeze(-1)
test_dates = dates[val_end:][:len(X_test)]

print("\nDataset sizes")
print("Train :", X_train.shape); print("Val   :", X_val.shape); print("Test  :", X_test.shape)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)


class LSTMGaussian(nn.Module):
    def __init__(self, lookback, hidden=LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.mean_head = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))
        self.scale_head = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))
        self.softplus = nn.Softplus()

    def encode(self, history):
        x = history.unsqueeze(-1)
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, history):
        h = self.encode(history)
        mean = self.mean_head(h)
        scale = self.softplus(self.scale_head(h)) + 1e-4
        eps = torch.randn_like(mean)
        return mean + scale * eps, mean, scale

    def sample(self, history, n_paths):
        h = self.encode(history)
        mean = self.mean_head(h)
        scale = self.softplus(self.scale_head(h)) + 1e-4
        samples, scales = [], []
        for _ in range(n_paths):
            eps = torch.randn_like(mean)
            samples.append(mean + scale * eps)
            scales.append(scale)
        return torch.stack(samples, dim=0), torch.stack(scales, dim=0)


model = LSTMGaussian(lookback=LOOKBACK, hidden=LSTM_HIDDEN).to(device)
print("\nModel:"); print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")

mean_tensor = torch.tensor(mean_y, dtype=torch.float32, device=device)
std_tensor = torch.tensor(std_y, dtype=torch.float32, device=device)


def to_rv(y_std):
    return torch.exp(y_std * std_tensor + mean_tensor)


def qlike_loss(actual, forecast):
    eps = 1e-8
    actual = torch.clamp(actual, min=eps); forecast = torch.clamp(forecast, min=eps)
    ratio = actual / forecast
    return torch.mean(ratio - torch.log(ratio) - 1.0)


def crps_ensemble(samples, target):
    M = samples.shape[0]
    term1 = torch.mean(torch.abs(samples - target.unsqueeze(0)))
    sorted_samples, _ = torch.sort(samples, dim=0)
    weights = torch.arange(1, M + 1, device=samples.device, dtype=samples.dtype)
    weights = (2.0 * weights - M - 1.0).view(M, 1, 1)
    pair_term = torch.sum(weights * sorted_samples, dim=0)
    return term1 - torch.mean(pair_term / (M ** 2))


def probabilistic_loss(samples_std, target_std):
    samples_rv = to_rv(samples_std); target_rv = to_rv(target_std)
    mean_rv = torch.mean(samples_rv, dim=0)
    loss_qlike = qlike_loss(target_rv, mean_rv)
    loss_crps = crps_ensemble(samples_rv, target_rv)
    rv_scale = torch.mean(target_rv) + 1e-8
    total = loss_qlike + LAMBDA_CRPS * (loss_crps / rv_scale)
    return total, loss_qlike, loss_crps / rv_scale


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)


@torch.no_grad()
def evaluate(model, loader, n_paths):
    model.eval()
    losses, qlikes, crpss = [], [], []
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        samples, _ = model.sample(Xb, n_paths)
        loss, q, c = probabilistic_loss(samples, Yb)
        losses.append(loss.item()); qlikes.append(q.item()); crpss.append(c.item())
    return np.mean(losses), np.mean(qlikes), np.mean(crpss)


best_val, best_state, best_epoch, patience_counter = np.inf, None, 0, 0
for epoch in range(EPOCHS):
    model.train()
    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad()
        samples, _ = model.sample(Xb, TRAIN_MC_PATHS)
        loss, q, c = probabilistic_loss(samples, Yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    val_loss, val_qlike, val_crps = evaluate(model, val_loader, VAL_MC_PATHS)
    scheduler.step(val_loss)

    if val_loss < best_val - MIN_DELTA:
        best_val, best_state, best_epoch, patience_counter = val_loss, copy.deepcopy(model.state_dict()), epoch + 1, 0
    else:
        patience_counter += 1

    if epoch == 0 or (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:4d} | Val = {val_loss:.6f} | QLIKE = {val_qlike:.6f} | CRPS = {val_crps:.6f} | LR = {lr:.2e}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

model.load_state_dict(best_state)
print("\nBest epoch:", best_epoch, "| Best val loss:", best_val)


@torch.no_grad()
def predict_distribution(model, X, n_paths=TEST_MC_PATHS, batch_size=128):
    model.eval()
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    all_samples, all_scales = [], []
    for (Xb,) in loader:
        Xb = Xb.to(device)
        samples, scales = model.sample(Xb, n_paths)
        all_samples.append(to_rv(samples).cpu()); all_scales.append(scales.cpu())
    return torch.cat(all_samples, dim=1).numpy().squeeze(-1), torch.cat(all_scales, dim=1).numpy().squeeze(-1)


forecast_samples, observation_scales = predict_distribution(model, X_test)
true_rv = np.exp(Y_test.numpy().squeeze() * std_y + mean_y)
mean_forecast = np.mean(forecast_samples, axis=0)
lower_95 = np.quantile(forecast_samples, 0.025, axis=0); upper_95 = np.quantile(forecast_samples, 0.975, axis=0)
lower_90 = np.quantile(forecast_samples, 0.05, axis=0); upper_90 = np.quantile(forecast_samples, 0.95, axis=0)


def rmse(a, f): return np.sqrt(np.mean((a - f) ** 2))
def mae(a, f): return np.mean(np.abs(a - f))
def mape(a, f): return 100 * np.mean(np.abs(a - f) / (a + 1e-8))
def qlike(a, f): r = a / (f + 1e-8); return np.mean(r - np.log(r + 1e-8) - 1.0)
def empirical_crps(samples, obs):
    M = samples.shape[0]
    first = np.mean(np.abs(samples - obs[None, :]), axis=0)
    sorted_s = np.sort(samples, axis=0)
    i = np.arange(1, M + 1)[:, None]
    second = np.sum((2 * i - M - 1) * sorted_s, axis=0) / (M ** 2)
    return np.mean(first - second)


coverage_95 = np.mean((true_rv >= lower_95) & (true_rv <= upper_95))
coverage_90 = np.mean((true_rv >= lower_90) & (true_rv <= upper_90))
pit = np.mean(forecast_samples <= true_rv[None, :], axis=0)

print("\n======================================\nLSTM BENCHMARK -- OUT-OF-SAMPLE RESULTS\n======================================")
print(f"RMSE              : {rmse(true_rv, mean_forecast):.6f}")
print(f"MAE               : {mae(true_rv, mean_forecast):.6f}")
print(f"MAPE              : {mape(true_rv, mean_forecast):.3f}%")
print(f"QLIKE             : {qlike(true_rv, mean_forecast):.6f}")
print(f"CRPS              : {empirical_crps(forecast_samples, true_rv):.6f}")
print(f"90% coverage      : {100*coverage_90:.2f}%")
print(f"95% coverage      : {100*coverage_95:.2f}%")
print(f"90% interval width: {np.mean(upper_90-lower_90):.6f}")
print(f"95% interval width: {np.mean(upper_95-lower_95):.6f}")

q50, q90 = np.quantile(true_rv, [0.50, 0.90])
low_mask = true_rv <= q50; medium_mask = (true_rv > q50) & (true_rv <= q90); high_mask = true_rv > q90
def regime_coverage(mask): return np.mean((true_rv[mask] >= lower_95[mask]) & (true_rv[mask] <= upper_95[mask]))
print("\n95% coverage by volatility regime")
print(f"Low volatility : {100*regime_coverage(low_mask):.2f}%")
print(f"Medium         : {100*regime_coverage(medium_mask):.2f}%")
print(f"High volatility: {100*regime_coverage(high_mask):.2f}%")

covid_mask_test = (test_dates >= "2020-02-15") & (test_dates <= "2020-04-30")
bear_mask_test = (test_dates >= "2022-01-01") & (test_dates <= "2022-12-31")
print("\nCrisis-window results (genuinely out-of-sample this time)")
if covid_mask_test.sum() > 0:
    print(f"COVID crash ({int(covid_mask_test.sum())} days): 95% coverage = {100*regime_coverage(covid_mask_test):.2f}%")
if bear_mask_test.sum() > 0:
    print(f"2022 bear market ({int(bear_mask_test.sum())} days): 95% coverage = {100*regime_coverage(bear_mask_test):.2f}%")

import os
os.makedirs("../results", exist_ok=True)
np.savez("../results/lstm_test_results.npz",
         true_rv=true_rv, mean_forecast=mean_forecast, forecast_samples=forecast_samples,
         lower_95=lower_95, upper_95=upper_95, lower_90=lower_90, upper_90=upper_90, pit=pit)
print("\nSaved ../results/lstm_test_results.npz")