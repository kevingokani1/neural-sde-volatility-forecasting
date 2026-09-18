from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SDE_PATH = RESULTS_DIR / "sde_test_results.npz"
LSTM_PATH = RESULTS_DIR / "lstm_test_results.npz"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_loader import load_rv_series

CRISIS_WINDOWS = {
    "COVID_crash": ("2020-02-15", "2020-04-30"),
    "2022_bear_market": ("2022-01-01", "2022-12-31"),
}


def qlike(a, f):
    ratio = a / (f + 1e-8)
    return float(np.mean(ratio - np.log(ratio + 1e-8) - 1.0))

def rmse(a, f):
    return float(np.sqrt(np.mean((a - f) ** 2)))

def diebold_mariano(loss_a, loss_b):
    d = loss_a - loss_b
    if d.std(ddof=1) == 0 or len(d) < 10:
        return float("nan"), float("nan")
    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def get_test_dates():
    _, _, T, dates = load_rv_series()
    val_end = int(0.85 * T)
    return dates[val_end:]


def bootstrap_ci(true_rv, forecast, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(true_rv)
    stats_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_boot[b] = metric_fn(true_rv[idx], forecast[idx])
    return float(np.percentile(stats_boot, 2.5)), float(np.percentile(stats_boot, 97.5))


def main():
    print("Loading results...")
    sde = dict(np.load(SDE_PATH))
    lstm = dict(np.load(LSTM_PATH))
    true_rv = sde["true_rv"]

    if not np.allclose(true_rv, lstm["true_rv"], rtol=1e-6):
        raise ValueError("true_rv mismatch between the two result files.")

    n_test = len(true_rv)
    print(f"Deep analysis on {n_test} out-of-sample days\n")

    print("=" * 60)
    print("1. BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 resamples)")
    print("=" * 60)
    boot_rows = []
    for name, results in [("Neural SDE", sde), ("LSTM", lstm)]:
        rmse_lo, rmse_hi = bootstrap_ci(true_rv, results["mean_forecast"], rmse)
        qlike_lo, qlike_hi = bootstrap_ci(true_rv, results["mean_forecast"], qlike)
        point_rmse = rmse(true_rv, results["mean_forecast"])
        point_qlike = qlike(true_rv, results["mean_forecast"])
        boot_rows.append({
            "model": name,
            "RMSE": point_rmse, "RMSE_95%_CI": f"[{rmse_lo:.6f}, {rmse_hi:.6f}]",
            "QLIKE": point_qlike, "QLIKE_95%_CI": f"[{qlike_lo:.4f}, {qlike_hi:.4f}]",
        })
        print(f"{name:12s} RMSE = {point_rmse:.6f}  95% CI [{rmse_lo:.6f}, {rmse_hi:.6f}]")
        print(f"{'':12s} QLIKE = {point_qlike:.4f}  95% CI [{qlike_lo:.4f}, {qlike_hi:.4f}]")
    pd.DataFrame(boot_rows).to_csv(RESULTS_DIR / "bootstrap_confidence_intervals.csv", index=False)

    print("\n" + "=" * 60)
    print("2. CRISIS-WINDOW VALIDATION (real historical dates)")
    print("=" * 60)
    try:
        test_dates = get_test_dates()
        aligned_dates = test_dates[-n_test:] if len(test_dates) >= n_test else None
    except Exception as e:
        aligned_dates = None
        print(f"Could not reconstruct test dates ({e}) -- skipping.")

    crisis_rows = []
    if aligned_dates is not None:
        dates_index = pd.DatetimeIndex(aligned_dates)
        for name, (start, end) in CRISIS_WINDOWS.items():
            mask = (dates_index >= start) & (dates_index <= end)
            n_days = int(mask.sum())
            if n_days == 0:
                print(f"{name}: not in test window (0 days)")
                continue
            for model_name, results in [("Neural SDE", sde), ("LSTM", lstm)]:
                cov95 = float(np.mean(
                    (true_rv[mask] >= results["lower_95"][mask]) &
                    (true_rv[mask] <= results["upper_95"][mask])
                ))
                q = qlike(true_rv[mask], results["mean_forecast"][mask])
                crisis_rows.append({"window": name, "n_days": n_days, "model": model_name,
                                     "coverage_95": 100 * cov95, "QLIKE": q})
                print(f"{name:18s} ({n_days:3d} days)  {model_name:12s}  "
                      f"coverage_95={100*cov95:5.1f}%  QLIKE={q:.4f}")
        if crisis_rows:
            pd.DataFrame(crisis_rows).to_csv(RESULTS_DIR / "crisis_window_validation.csv", index=False)
        else:
            print("Neither crisis window falls inside your test set (likely inside train/val).")

    print("\n" + "=" * 60)
    print("3. REGIME-SPECIFIC DIEBOLD-MARIANO TESTS")
    print("=" * 60)
    q50, q90 = np.quantile(true_rv, [0.50, 0.90])
    regimes = {
        "Low volatility": true_rv <= q50,
        "Medium volatility": (true_rv > q50) & (true_rv <= q90),
        "High volatility": true_rv > q90,
    }
    dm_rows = []
    for regime_name, mask in regimes.items():
        loss_sde = (true_rv[mask] - sde["mean_forecast"][mask]) ** 2
        loss_lstm = (true_rv[mask] - lstm["mean_forecast"][mask]) ** 2
        dm_stat, p_val = diebold_mariano(loss_sde, loss_lstm)
        dm_rows.append({"regime": regime_name, "n_days": int(mask.sum()), "dm_stat": dm_stat, "p_value": p_val})
        sig = "significant" if (p_val == p_val and p_val < 0.05) else "not significant"
        print(f"{regime_name:20s} (n={int(mask.sum()):3d})  DM stat={dm_stat:.3f}  p={p_val:.3f}  ({sig})")
    pd.DataFrame(dm_rows).to_csv(RESULTS_DIR / "regime_dm_tests.csv", index=False)

    print("\n" + "=" * 60)
    print("4. CALIBRATION TEST (Kolmogorov-Smirnov, PIT vs Uniform[0,1])")
    print("=" * 60)
    ks_rows = []
    for name, results in [("Neural SDE", sde), ("LSTM", lstm)]:
        ks_stat, ks_p = stats.kstest(results["pit"], "uniform")
        ks_rows.append({"model": name, "ks_stat": float(ks_stat), "p_value": float(ks_p)})
        verdict = "well-calibrated" if ks_p >= 0.05 else "miscalibrated"
        print(f"{name:12s} KS stat = {ks_stat:.4f}, p = {ks_p:.4f}  -> {verdict}")
    pd.DataFrame(ks_rows).to_csv(RESULTS_DIR / "calibration_ks_test.csv", index=False)

    print("\n" + "=" * 60)
    print("5. FORECAST VISUALIZATION")
    print("=" * 60)
    x = np.arange(n_test)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, true_rv, color="black", linewidth=1, label="True RV")
    axes[0].plot(x, sde["mean_forecast"], color="steelblue", linewidth=1, label="Neural SDE forecast")
    axes[0].fill_between(x, sde["lower_95"], sde["upper_95"], color="steelblue", alpha=0.2, label="SDE 95% interval")
    axes[0].set_title("Neural SDE: forecast vs actual (test set)")
    axes[0].legend(fontsize=8)

    axes[1].plot(x, true_rv, color="black", linewidth=1, label="True RV")
    axes[1].plot(x, lstm["mean_forecast"], color="darkorange", linewidth=1, label="LSTM forecast")
    axes[1].fill_between(x, lstm["lower_95"], lstm["upper_95"], color="darkorange", alpha=0.2, label="LSTM 95% interval")
    axes[1].set_title("LSTM: forecast vs actual (test set)")
    axes[1].set_xlabel("Test day index")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'forecast_vs_actual.png'}")

    print("\n" + "=" * 60)
    print("ALL DEEP-ANALYSIS FILES SAVED TO results/")
    print("=" * 60)


if __name__ == "__main__":
    main()