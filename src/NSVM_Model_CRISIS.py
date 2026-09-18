# ============================================================
# Neural SDE v3 for Realized-Volatility Forecasting
#
# Latent dynamics:
#
#   dX_t = mu_theta(X_t,t) dt
#        + sigma_theta(X_t,t) dW_t
#
# Observation model:
#
#   Y_{t+1} = m_phi(X_{t+1})
#           + tau_phi(X_{t+1}) epsilon
#
# where Y is standardized log-realized volatility.
#
# Training:
#
#   Loss = QLIKE + lambda_CRPS * normalized CRPS
#
# Evaluation:
#
#   RMSE, MAE, MAPE, QLIKE, CRPS
#   90% and 95% coverage
#   interval width
#   PIT calibration
#
# ============================================================

import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

LOOKBACK = 20
HORIZON = 1

LATENT_DIM = 8
N_STEPS = 12

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


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 3. Data
#
# Synthetic example.
#
# Replace ONLY this section with real S&P 500 realized
# volatility when moving to the empirical experiment.
# ============================================================

from data_loader import load_rv_series

log_rv, rv, T, dates = load_rv_series()


# ============================================================
# 4. Chronological train / validation / test split
#
# CALENDAR-BASED (fixed dates), not percentage-based -- this is
# the crisis-window fix. Train through 2017, validate on 2018-19
# (calm), test from 2020 onward. This puts BOTH the COVID-19
# crash and the 2022 bear market genuinely out-of-sample, in the
# test set, rather than inside training as the earlier
# percentage-based (70/15/15) split did.
# ============================================================

from data_loader import calendar_split_indices

train_end, val_end = calendar_split_indices(dates)
print(f"Train ends: {dates[train_end-1].date()}  |  "
      f"Val ends: {dates[val_end-1].date()}  |  "
      f"Test: {dates[val_end].date()} to {dates[-1].date()}")

# ============================================================
# 5. Standardization
#
# Training data ONLY.
# ============================================================

mean_y = log_rv[:train_end].mean()
std_y = log_rv[:train_end].std()

y = (
    (log_rv - mean_y)
    / std_y
)


# ============================================================
# 6. Construct rolling windows
# ============================================================

def make_sequences(
    series,
    lookback,
    horizon
):

    X = []
    Y = []
    indices = []

    end = (
        len(series)
        - lookback
        - horizon
        + 1
    )

    for i in range(end):

        history = series[
            i:i + lookback
        ]

        target_index = (
            i
            + lookback
            + horizon
            - 1
        )

        target = series[
            target_index
        ]

        X.append(history)
        Y.append(target)
        indices.append(target_index)

    return (
        np.asarray(X),
        np.asarray(Y),
        np.asarray(indices)
    )


X_all, Y_all, target_idx = make_sequences(
    y,
    LOOKBACK,
    HORIZON
)


# ============================================================
# 7. Chronological sequence split
# ============================================================

train_mask = (
    target_idx < train_end
)

val_mask = (
    (target_idx >= train_end)
    &
    (target_idx < val_end)
)

test_mask = (
    target_idx >= val_end
)


def tensor(x):

    return torch.tensor(
        x,
        dtype=torch.float32
    )


X_train = tensor(
    X_all[train_mask]
)

Y_train = tensor(
    Y_all[train_mask]
).unsqueeze(-1)


X_val = tensor(
    X_all[val_mask]
)

Y_val = tensor(
    Y_all[val_mask]
).unsqueeze(-1)


X_test = tensor(
    X_all[test_mask]
)

Y_test = tensor(
    Y_all[test_mask]
).unsqueeze(-1)


print("\nDataset sizes")

print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)


# ============================================================
# 8. DataLoaders
# ============================================================

train_loader = DataLoader(

    TensorDataset(
        X_train,
        Y_train
    ),

    batch_size=BATCH_SIZE,

    shuffle=True
)


val_loader = DataLoader(

    TensorDataset(
        X_val,
        Y_val
    ),

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# 9. Generic MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# 10. Encoder
#
# Historical volatility:
#
# (y_{t-L+1},...,y_t)
#
#        |
#        v
#
# Initial latent state X_0
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        lookback,
        latent_dim
    ):

        super().__init__()

        self.net = MLP(
            lookback,
            latent_dim
        )

    def forward(self, history):

        return self.net(history)


# ============================================================
# 11. Drift network
#
# mu_theta(X,t)
# ============================================================

class DriftNetwork(nn.Module):

    def __init__(
        self,
        latent_dim
    ):

        super().__init__()

        self.net = MLP(
            latent_dim + 1,
            latent_dim
        )

    def forward(
        self,
        x,
        t
    ):

        z = torch.cat(
            [x, t],
            dim=-1
        )

        return self.net(z)


# ============================================================
# 12. Diffusion network
#
# sigma_theta(X,t)
#
# Softplus ensures positivity.
#
# IMPORTANT:
# No hard clamp.
# ============================================================

class DiffusionNetwork(nn.Module):

    def __init__(
        self,
        latent_dim
    ):

        super().__init__()

        self.net = MLP(
            latent_dim + 1,
            latent_dim
        )

        self.softplus = nn.Softplus()

    def forward(
        self,
        x,
        t
    ):

        z = torch.cat(
            [x, t],
            dim=-1
        )

        raw_sigma = self.net(z)

        sigma = (
            self.softplus(raw_sigma)
            + 1e-4
        )

        return sigma


# ============================================================
# 13. Observation model
#
# Given latent X_T:
#
#   mean = m_phi(X_T)
#   scale = tau_phi(X_T) > 0
#
# Then
#
#   Y = mean + scale * epsilon
#
# ============================================================

class ObservationModel(nn.Module):

    def __init__(
        self,
        latent_dim
    ):

        super().__init__()

        self.mean_net = nn.Sequential(

            nn.Linear(
                latent_dim,
                32
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                1
            )
        )

        self.scale_net = nn.Sequential(

            nn.Linear(
                latent_dim,
                32
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                1
            )
        )

        self.softplus = nn.Softplus()

    def forward(self, x):

        mean = self.mean_net(x)

        scale = (
            self.softplus(
                self.scale_net(x)
            )
            + 1e-4
        )

        return mean, scale


# ============================================================
# 14. Complete Neural SDE
# ============================================================

class NeuralSDE(nn.Module):

    def __init__(
        self,
        lookback,
        latent_dim=8,
        n_steps=12
    ):

        super().__init__()

        self.latent_dim = latent_dim
        self.n_steps = n_steps

        self.encoder = Encoder(
            lookback,
            latent_dim
        )

        self.drift = DriftNetwork(
            latent_dim
        )

        self.diffusion = DiffusionNetwork(
            latent_dim
        )

        self.observation = ObservationModel(
            latent_dim
        )


    # --------------------------------------------------------
    # Euler-Maruyama
    # --------------------------------------------------------

    def solve_sde(
        self,
        x0
    ):

        dt = (
            1.0
            / self.n_steps
        )

        sqrt_dt = np.sqrt(dt)

        x = x0

        batch_size = (
            x.shape[0]
        )

        for n in range(
            self.n_steps
        ):

            t_value = (
                n * dt
            )

            t = torch.full(

                (
                    batch_size,
                    1
                ),

                t_value,

                dtype=x.dtype,

                device=x.device
            )

            mu = self.drift(
                x,
                t
            )

            sigma = self.diffusion(
                x,
                t
            )

            dW = (
                torch.randn_like(x)
                * sqrt_dt
            )

            x = (
                x
                + mu * dt
                + sigma * dW
            )

        return x


    # --------------------------------------------------------
    # One stochastic forecast
    # --------------------------------------------------------

    def forward(
        self,
        history
    ):

        x0 = self.encoder(
            history
        )

        xT = self.solve_sde(
            x0
        )

        mean, scale = (
            self.observation(xT)
        )

        epsilon = (
            torch.randn_like(mean)
        )

        y_sample = (
            mean
            + scale * epsilon
        )

        return (
            y_sample,
            mean,
            scale
        )


    # --------------------------------------------------------
    # Monte Carlo distribution
    # --------------------------------------------------------

    def sample(
        self,
        history,
        n_paths
    ):

        samples = []

        latent_scales = []
        observation_scales = []

        for _ in range(
            n_paths
        ):

            x0 = self.encoder(
                history
            )

            xT = self.solve_sde(
                x0
            )

            mean, scale = (
                self.observation(xT)
            )

            epsilon = (
                torch.randn_like(mean)
            )

            y = (
                mean
                + scale * epsilon
            )

            samples.append(y)

            observation_scales.append(
                scale
            )

        return (
            torch.stack(
                samples,
                dim=0
            ),
            torch.stack(
                observation_scales,
                dim=0
            )
        )


# ============================================================
# 15. Initialise model
# ============================================================

model = NeuralSDE(

    lookback=LOOKBACK,

    latent_dim=LATENT_DIM,

    n_steps=N_STEPS

).to(device)


print("\nModel:")
print(model)


# ============================================================
# 16. Transform standardized log-RV -> RV
# ============================================================

mean_tensor = torch.tensor(
    mean_y,
    dtype=torch.float32,
    device=device
)

std_tensor = torch.tensor(
    std_y,
    dtype=torch.float32,
    device=device
)


def to_rv(y_standardized):

    log_rv_value = (
        y_standardized
        * std_tensor
        + mean_tensor
    )

    return torch.exp(
        log_rv_value
    )


# ============================================================
# 17. QLIKE
# ============================================================

def qlike_loss(
    actual,
    forecast
):

    eps = 1e-8

    actual = torch.clamp(
        actual,
        min=eps
    )

    forecast = torch.clamp(
        forecast,
        min=eps
    )

    ratio = (
        actual
        / forecast
    )

    return torch.mean(

        ratio
        - torch.log(ratio)
        - 1.0
    )


# ============================================================
# 18. Efficient ensemble CRPS
#
# Exact O(M log M) ensemble formula.
#
# Avoids the M x M matrix used previously.
# ============================================================

def crps_ensemble(
    samples,
    target
):

    # samples:
    #
    # [M, batch, 1]

    M = samples.shape[0]

    term1 = torch.mean(
        torch.abs(
            samples
            - target.unsqueeze(0)
        )
    )

    sorted_samples, _ = torch.sort(
        samples,
        dim=0
    )

    weights = torch.arange(
        1,
        M + 1,
        device=samples.device,
        dtype=samples.dtype
    )

    weights = (
        2.0 * weights
        - M
        - 1.0
    )

    weights = weights.view(
        M,
        1,
        1
    )

    pair_term = torch.sum(
        weights
        * sorted_samples,
        dim=0
    )

    term2 = torch.mean(
        pair_term
        / (M ** 2)
    )

    return (
        term1
        - term2
    )


# ============================================================
# 19. Probabilistic objective
# ============================================================

def probabilistic_loss(
    samples_standardized,
    target_standardized
):

    samples_rv = to_rv(
        samples_standardized
    )

    target_rv = to_rv(
        target_standardized
    )

    # Conditional mean

    mean_rv = torch.mean(
        samples_rv,
        dim=0
    )

    # Point forecast quality

    loss_qlike = qlike_loss(
        target_rv,
        mean_rv
    )

    # Distribution quality

    loss_crps = crps_ensemble(
        samples_rv,
        target_rv
    )

    # Scale normalization

    rv_scale = (
        torch.mean(target_rv)
        + 1e-8
    )

    normalized_crps = (
        loss_crps
        / rv_scale
    )

    total = (
        loss_qlike
        + LAMBDA_CRPS
        * normalized_crps
    )

    return (
        total,
        loss_qlike,
        normalized_crps
    )


# ============================================================
# 20. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=10
    )
)


# ============================================================
# 21. Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    n_paths
):

    model.eval()

    losses = []
    qlikes = []
    crps_values = []

    for X_batch, Y_batch in loader:

        X_batch = (
            X_batch.to(device)
        )

        Y_batch = (
            Y_batch.to(device)
        )

        samples, _ = model.sample(
            X_batch,
            n_paths
        )

        loss, q, c = (
            probabilistic_loss(
                samples,
                Y_batch
            )
        )

        losses.append(
            loss.item()
        )

        qlikes.append(
            q.item()
        )

        crps_values.append(
            c.item()
        )

    return (
        np.mean(losses),
        np.mean(qlikes),
        np.mean(crps_values)
    )


# ============================================================
# 22. Training
# ============================================================

train_history = []
val_history = []

val_qlike_history = []
val_crps_history = []

best_val = np.inf
best_state = None
best_epoch = 0

patience_counter = 0


for epoch in range(EPOCHS):

    model.train()

    batch_losses = []

    for X_batch, Y_batch in train_loader:

        X_batch = (
            X_batch.to(device)
        )

        Y_batch = (
            Y_batch.to(device)
        )

        optimizer.zero_grad()

        samples, _ = model.sample(
            X_batch,
            TRAIN_MC_PATHS
        )

        loss, q, c = (
            probabilistic_loss(
                samples,
                Y_batch
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_losses.append(
            loss.item()
        )


    train_loss = np.mean(
        batch_losses
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        val_qlike,
        val_crps
    ) = evaluate(

        model,

        val_loader,

        VAL_MC_PATHS
    )


    scheduler.step(
        val_loss
    )


    train_history.append(
        train_loss
    )

    val_history.append(
        val_loss
    )

    val_qlike_history.append(
        val_qlike
    )

    val_crps_history.append(
        val_crps
    )


    # --------------------------------------------------------
    # Early stopping with minimum meaningful improvement
    # --------------------------------------------------------

    if (
        val_loss
        <
        best_val - MIN_DELTA
    ):

        best_val = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = (
            epoch + 1
        )

        patience_counter = 0

    else:

        patience_counter += 1


    if (
        epoch == 0
        or
        (epoch + 1) % 10 == 0
    ):

        lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(

            f"Epoch {epoch + 1:4d} | "

            f"Train = {train_loss:.6f} | "

            f"Val = {val_loss:.6f} | "

            f"QLIKE = {val_qlike:.6f} | "

            f"CRPS = {val_crps:.6f} | "

            f"LR = {lr:.2e}"
        )


    if (
        patience_counter
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            f"\nEarly stopping "
            f"at epoch {epoch + 1}"
        )

        break


# ============================================================
# 23. Restore best model
# ============================================================

model.load_state_dict(
    best_state
)

print(
    "\nBest epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    best_val
)


# ============================================================
# 24. Training curves
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    train_history,
    label="Training loss"
)

plt.plot(
    val_history,
    label="Validation loss"
)

plt.axvline(
    best_epoch - 1,
    linestyle="--",
    label="Best epoch"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Combined loss"
)

plt.title(
    "Neural SDE probabilistic training"
)

plt.legend()

plt.tight_layout()

#plt.show()

plt.savefig("training_curves.png", dpi=150)
plt.close()


# ============================================================
# 25. Test prediction
# ============================================================

@torch.no_grad()
def predict_distribution(
    model,
    X,
    n_paths=200,
    batch_size=128
):

    model.eval()

    loader = DataLoader(

        TensorDataset(X),

        batch_size=batch_size,

        shuffle=False
    )

    all_samples = []
    all_scales = []

    for (X_batch,) in loader:

        X_batch = (
            X_batch.to(device)
        )

        samples, scales = model.sample(
            X_batch,
            n_paths
        )

        samples_rv = to_rv(
            samples
        )

        all_samples.append(
            samples_rv.cpu()
        )

        all_scales.append(
            scales.cpu()
        )

    samples = torch.cat(
        all_samples,
        dim=1
    )

    scales = torch.cat(
        all_scales,
        dim=1
    )

    return (
        samples.numpy().squeeze(-1),
        scales.numpy().squeeze(-1)
    )


forecast_samples, observation_scales = (
    predict_distribution(

        model,

        X_test,

        TEST_MC_PATHS
    )
)


# ============================================================
# 26. Actual RV
# ============================================================

true_standardized = (
    Y_test.numpy().squeeze()
)

true_log_rv = (
    true_standardized
    * std_y
    + mean_y
)

true_rv = np.exp(
    true_log_rv
)


# ============================================================
# 27. Predictive statistics
# ============================================================

mean_forecast = np.mean(
    forecast_samples,
    axis=0
)

median_forecast = np.median(
    forecast_samples,
    axis=0
)


lower_95 = np.quantile(
    forecast_samples,
    0.025,
    axis=0
)

upper_95 = np.quantile(
    forecast_samples,
    0.975,
    axis=0
)


lower_90 = np.quantile(
    forecast_samples,
    0.05,
    axis=0
)

upper_90 = np.quantile(
    forecast_samples,
    0.95,
    axis=0
)


# ============================================================
# 28. Point metrics
# ============================================================

def rmse(
    actual,
    forecast
):

    return np.sqrt(
        np.mean(
            (actual - forecast) ** 2
        )
    )


def mae(
    actual,
    forecast
):

    return np.mean(
        np.abs(
            actual - forecast
        )
    )


def mape(
    actual,
    forecast
):

    return (
        100
        * np.mean(
            np.abs(
                actual - forecast
            )
            /
            (
                actual + 1e-8
            )
        )
    )


def qlike(
    actual,
    forecast
):

    ratio = (
        actual
        /
        (
            forecast + 1e-8
        )
    )

    return np.mean(
        ratio
        - np.log(
            ratio + 1e-8
        )
        - 1.0
    )


# ============================================================
# 29. Exact empirical CRPS
#
# Uses sorted ensemble formula.
# ============================================================

def empirical_crps(
    samples,
    observations
):

    M = samples.shape[0]

    first = np.mean(
        np.abs(
            samples
            - observations[None, :]
        ),
        axis=0
    )

    sorted_samples = np.sort(
        samples,
        axis=0
    )

    i = np.arange(
        1,
        M + 1
    )[:, None]

    weights = (
        2 * i
        - M
        - 1
    )

    second = np.sum(
        weights
        * sorted_samples,
        axis=0
    ) / (M ** 2)

    return np.mean(
        first - second
    )


# ============================================================
# 30. Coverage
# ============================================================

coverage_95 = np.mean(

    (true_rv >= lower_95)
    &
    (true_rv <= upper_95)
)


coverage_90 = np.mean(

    (true_rv >= lower_90)
    &
    (true_rv <= upper_90)
)


width_95 = np.mean(
    upper_95
    - lower_95
)

width_90 = np.mean(
    upper_90
    - lower_90
)


# ============================================================
# 31. PIT
# ============================================================

pit = np.mean(

    forecast_samples
    <= true_rv[None, :],

    axis=0
)


# ============================================================
# 32. Results
# ============================================================

print(
    "\n======================================"
)

print(
    "OUT-OF-SAMPLE RESULTS"
)

print(
    "======================================"
)


print(
    f"RMSE              : "
    f"{rmse(true_rv, mean_forecast):.6f}"
)

print(
    f"MAE               : "
    f"{mae(true_rv, mean_forecast):.6f}"
)

print(
    f"MAPE              : "
    f"{mape(true_rv, mean_forecast):.3f}%"
)

print(
    f"QLIKE             : "
    f"{qlike(true_rv, mean_forecast):.6f}"
)

print(
    f"CRPS              : "
    f"{empirical_crps(forecast_samples, true_rv):.6f}"
)

print(
    f"90% coverage      : "
    f"{100 * coverage_90:.2f}%"
)

print(
    f"95% coverage      : "
    f"{100 * coverage_95:.2f}%"
)

print(
    f"90% interval width: "
    f"{width_90:.6f}"
)

print(
    f"95% interval width: "
    f"{width_95:.6f}"
)


# ============================================================
# 33. Observation uncertainty diagnostics
# ============================================================

print(
    "\nObservation uncertainty"
)

print(
    "Mean tau:",
    observation_scales.mean()
)

print(
    "Std tau :",
    observation_scales.std()
)

print(
    "Min tau :",
    observation_scales.min()
)

print(
    "Max tau :",
    observation_scales.max()
)


# ============================================================
# 34. Latent diffusion diagnostics
# ============================================================

@torch.no_grad()
def diffusion_diagnostics(
    model,
    X
):

    model.eval()

    loader = DataLoader(
        TensorDataset(X),
        batch_size=256,
        shuffle=False
    )

    sigmas = []

    for (X_batch,) in loader:

        X_batch = (
            X_batch.to(device)
        )

        x0 = model.encoder(
            X_batch
        )

        t0 = torch.zeros(
            (
                X_batch.shape[0],
                1
            ),
            device=device
        )

        sigma = model.diffusion(
            x0,
            t0
        )

        sigmas.append(
            sigma.cpu()
        )

    return torch.cat(
        sigmas,
        dim=0
    ).numpy()


sigma_values = diffusion_diagnostics(
    model,
    X_test
)


print(
    "\nLatent diffusion"
)

print(
    "Mean sigma:",
    sigma_values.mean()
)

print(
    "Std sigma :",
    sigma_values.std()
)

print(
    "Min sigma :",
    sigma_values.min()
)

print(
    "Max sigma :",
    sigma_values.max()
)


# ============================================================
# 35. Forecast plot
# ============================================================

N_PLOT = min(
    300,
    len(true_rv)
)

x_axis = np.arange(
    N_PLOT
)


plt.figure(
    figsize=(13, 6)
)

plt.plot(
    x_axis,
    true_rv[:N_PLOT],
    label="Observed RV"
)

plt.plot(
    x_axis,
    mean_forecast[:N_PLOT],
    label="Neural SDE mean forecast"
)

plt.fill_between(
    x_axis,
    lower_95[:N_PLOT],
    upper_95[:N_PLOT],
    alpha=0.20,
    label="95% predictive interval"
)

plt.xlabel(
    "Out-of-sample day"
)

plt.ylabel(
    "Realized volatility"
)

plt.title(
    "Neural SDE realized-volatility forecast"
)

plt.legend()

plt.tight_layout()

#plt.show()


plt.savefig("forecast_plot.png", dpi=150)
plt.close()


# ============================================================
# 36. Predictive distribution
# ============================================================

OBSERVATION = 20

distribution = (
    forecast_samples[
        :,
        OBSERVATION
    ]
)

observed = (
    true_rv[
        OBSERVATION
    ]
)


plt.figure(
    figsize=(9, 5)
)

plt.hist(
    distribution,
    bins=40,
    density=True
)

plt.axvline(
    observed,
    linestyle="--",
    linewidth=2,
    label="Observed RV"
)

plt.axvline(
    distribution.mean(),
    linestyle=":",
    linewidth=2,
    label="Mean forecast"
)

plt.xlabel(
    "Forecast realized volatility"
)

plt.ylabel(
    "Density"
)

plt.title(
    "Neural SDE predictive distribution"
)

plt.legend()

plt.tight_layout()

#plt.show()


plt.savefig("predictive_distribution.png", dpi=150)
plt.close()


# ============================================================
# 37. PIT histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    pit,
    bins=10,
    density=True
)

# Uniform density reference
plt.axhline(
    1.0,
    linestyle="--",
    label="Uniform reference"
)

plt.xlabel(
    "PIT"
)

plt.ylabel(
    "Density"
)

plt.title(
    "Probability Integral Transform"
)

plt.legend()

plt.tight_layout()

#plt.show()



plt.savefig("pit_histogram.png", dpi=150)
plt.close()

# ============================================================
# 38. Coverage by volatility regime
#
# Useful for checking whether uncertainty calibration fails
# specifically during high-volatility periods.
# ============================================================

q50 = np.quantile(
    true_rv,
    0.50
)

q90 = np.quantile(
    true_rv,
    0.90
)


low_mask = (
    true_rv <= q50
)

medium_mask = (
    (true_rv > q50)
    &
    (true_rv <= q90)
)

high_mask = (
    true_rv > q90
)


def regime_coverage(
    mask
):

    return np.mean(

        (
            true_rv[mask]
            >= lower_95[mask]
        )

        &

        (
            true_rv[mask]
            <= upper_95[mask]
        )
    )


print(
    "\n95% coverage by volatility regime"
)

print(
    f"Low volatility : "
    f"{100 * regime_coverage(low_mask):.2f}%"
)

print(
    f"Medium         : "
    f"{100 * regime_coverage(medium_mask):.2f}%"
)

print(
    f"High volatility: "
    f"{100 * regime_coverage(high_mask):.2f}%"
)


# ============================================================
# Crisis-window coverage check
# ============================================================

covid_mask_test = (dates[val_end:] >= "2020-02-15") & (dates[val_end:] <= "2020-04-30")
bear_mask_test = (dates[val_end:] >= "2022-01-01") & (dates[val_end:] <= "2022-12-31")

print("\nCrisis-window results (genuinely out-of-sample this time)")
if covid_mask_test.sum() > 0:
    print(f"COVID crash ({int(covid_mask_test.sum())} days): "
          f"95% coverage = {100 * regime_coverage(covid_mask_test):.2f}%")
if bear_mask_test.sum() > 0:
    print(f"2022 bear market ({int(bear_mask_test.sum())} days): "
          f"95% coverage = {100 * regime_coverage(bear_mask_test):.2f}%")



import os
os.makedirs("../results", exist_ok=True)
np.savez(
    "../results/sde_test_results.npz",
    true_rv=true_rv,
    mean_forecast=mean_forecast,
    forecast_samples=forecast_samples,
    lower_95=lower_95, upper_95=upper_95,
    lower_90=lower_90, upper_90=upper_90,
    pit=pit,
)
print("\nSaved ../results/sde_test_results.npz")