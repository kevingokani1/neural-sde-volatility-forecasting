from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SDE_PATH = RESULTS_DIR / "sde_test_results.npz"
LSTM_PATH = RESULTS_DIR / "lstm_test_results.npz"


def load_results(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return dict(np.load(path))


def rmse(a, f):
    return float(np.sqrt(np.mean((a - f) ** 2)))

def mae(a, f):
    return float(np.mean(np.abs(a - f)))

def mape(a, f):
    return float(100 * np.mean(np.abs(a - f) / (a + 1e-8)))

def qlike(a, f):
    ratio = a / (f + 1e-8)
    return float(np.mean(ratio - np.log(ratio + 1e-8) - 1.0))

def empirical_crps(samples, observations):
    M = samples.shape[0]
    first = np.mean(np.abs(samples - observations[None, :]), axis=0)
    sorted_samples = np.sort(samples, axis=0)
    i = np.arange(1, M + 1)[:, None]
    weights = 2 * i - M - 1
    second = np.sum(weights * sorted_samples, axis=0) / (M ** 2)
    return float(np.mean(first - second))

def diebold_mariano(loss_a, loss_b):
    d = loss_a - loss_b
    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def build_metrics(results, true_rv):
    mean_forecast = results["mean_forecast"]
    samples = results["forecast_samples"]
    coverage_90 = float(np.mean((true_rv >= results["lower_90"]) & (true_rv <= results["upper_90"])))
    coverage_95 = float(np.mean((true_rv >= results["lower_95"]) & (true_rv <= results["upper_95"])))
    width_90 = float(np.mean(results["upper_90"] - results["lower_90"]))
    width_95 = float(np.mean(results["upper_95"] - results["lower_95"]))
    return {
        "RMSE": rmse(true_rv, mean_forecast),
        "MAE": mae(true_rv, mean_forecast),
        "MAPE (%)": mape(true_rv, mean_forecast),
        "QLIKE": qlike(true_rv, mean_forecast),
        "CRPS": empirical_crps(samples, true_rv),
        "90% coverage (%)": 100 * coverage_90,
        "95% coverage (%)": 100 * coverage_95,
        "90% interval width": width_90,
        "95% interval width": width_95,
    }


def main():
    print("Loading result files...")
    sde = load_results(SDE_PATH)
    lstm = load_results(LSTM_PATH)
    print("Loaded both files.")

    true_rv = sde["true_rv"]

    sde_metrics = build_metrics(sde, true_rv)
    lstm_metrics = build_metrics(lstm, true_rv)

    table = pd.DataFrame({"Neural SDE": sde_metrics, "LSTM (Gaussian)": lstm_metrics})
    table.to_csv(RESULTS_DIR / "final_comparison_table.csv")

    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(table.to_string())

    loss_sde = (true_rv - sde["mean_forecast"]) ** 2
    loss_lstm = (true_rv - lstm["mean_forecast"]) ** 2
    dm_stat, p_value = diebold_mariano(loss_sde, loss_lstm)
    print(f"\nDiebold-Mariano test: stat = {dm_stat:.4f}, p = {p_value:.4f}")

    fig, axes = plt.subplots(1, len(table.index), figsize=(4 * len(table.index), 4))
    for ax, metric in zip(axes, table.index):
        table.loc[metric].plot(kind="bar", ax=ax, color=["steelblue", "darkorange"])
        ax.set_title(metric, fontsize=10)
        ax.set_xticklabels(table.columns, rotation=0)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "final_comparison_figure.png", dpi=150)
    plt.close(fig)

    print("\nDone. Saved final_comparison_table.csv and final_comparison_figure.png")


print("SCRIPT STARTED")
main()
print("SCRIPT FINISHED")