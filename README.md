# Comparative Analysis of Neural SDE and Deep Learning Models for Financial Volatility Forecasting

## Overview

This repository contains the implementation and empirical analysis for an MSc Finance Data Science dissertation investigating the use of **Neural Stochastic Differential Equations (Neural SDEs)** for financial volatility forecasting.

The study compares a Neural SDE forecasting framework against a recurrent deep-learning benchmark based on a **Long Short-Term Memory (LSTM)** network. The models produce probabilistic volatility forecasts and are evaluated using both point-forecast accuracy and distributional/calibration measures.

The research focuses on whether the continuous-time stochastic structure of Neural SDEs provides useful advantages over conventional deep-learning approaches for forecasting financial volatility.       

---

## Research Objective

The main objective is to conduct a comparative empirical analysis of:

- Neural SDE-based volatility forecasting
- LSTM-based volatility forecasting
- Point forecast accuracy
- Probabilistic forecast quality
- Prediction interval coverage
- Distributional calibration
- Performance across different market regimes

The analysis uses realised variance as the volatility forecasting target.

---

## Methodology

### Neural SDE

The Neural SDE model represents the latent dynamics of volatility using a continuous-time stochastic process:

$$
dX_t = f_\theta(X_t)\,dt + g_\theta(X_t)\,dW_t
$$

where:

- $X_t$ represents the latent state
- $f_\theta$ is the neural drift function
- $g_\theta$ is the neural diffusion function
- $W_t$ is a Wiener process

The implementation uses:

- 8-dimensional latent state
- 20-observation lookback window
- Neural encoder for latent-state initialisation
- Two-layer MLP drift network
- Two-layer MLP diffusion network
- 64 hidden units
- Tanh activations
- Softplus diffusion output
- Euler-Maruyama numerical integration
- 12 integration steps
- 200 Monte Carlo paths
- Gaussian probabilistic observation model

### LSTM Benchmark

The benchmark model uses an LSTM architecture to capture temporal dependencies in realised volatility.

The principal configuration uses:

- 32-dimensional hidden state
- LSTM recurrent layer
- Gaussian predictive distribution
- Separate predictive mean and scale components

---

## Training

The models are trained using a combined probabilistic objective:

$$
\mathcal{L} =
\text{QLIKE} + 0.5 \times \text{CRPS}
$$

where:

- **QLIKE** evaluates volatility forecast accuracy
- **CRPS** evaluates probabilistic forecast quality

Training uses:

- Adam optimisation
- Learning rate: $10^{-3}$
- Gradient clipping
- Early stopping
- Fixed random seed

---

## Data and Experimental Design

The empirical analysis uses realised variance data for financial-market volatility forecasting.

The chronological data split is:

| Dataset | Period |
|---|---|
| Training | 2009–2017 |
| Validation | 2018–2019 |
| Test | 2020–2026 |

A chronological split is used to avoid using future information when estimating the forecasting models.

The raw dataset is **not included in this repository**. This repository contains the modelling, evaluation and analysis code together with the generated research outputs.

---

## Evaluation Metrics

The models are evaluated using:

- RMSE
- MAE
- MAPE
- QLIKE
- CRPS
- 90% prediction-interval coverage
- 95% prediction-interval coverage
- Prediction-interval width
- Probability Integral Transform (PIT)
- Kolmogorov-Smirnov calibration test
- Diebold-Mariano test
- Bootstrap confidence intervals
- Regime/crisis analysis

---

## Main Results

The final out-of-sample comparison produced the following results:

| Metric | Neural SDE | LSTM |
|---|---:|---:|
| RMSE | 0.000213 | 0.000218 |
| MAE | 0.000055 | 0.000053 |
| MAPE (%) | 80.9769 | 81.4437 |
| QLIKE | 0.251889 | 0.260971 |
| CRPS | 0.000041 | 0.000039 |
| 90% Coverage (%) | 69.5719 | 84.7095 |
| 95% Coverage (%) | 77.8287 | 90.3670 |
| 90% Interval Width | 0.000122 | 0.000182 |
| 95% Interval Width | 0.000147 | 0.000240 |

The Diebold-Mariano comparison produced:

- Test statistic: **-0.1220**
- p-value: **0.9029**

These results indicate that the difference in predictive losses was not statistically significant according to this test.

The results also show a distinction between point-forecast performance and probabilistic calibration. The Neural SDE produced narrower prediction intervals, while the LSTM achieved higher empirical coverage in the reported test period.

---

## Calibration Analysis

Probability Integral Transform (PIT) diagnostics were used to assess whether the predictive distributions were consistent with the observed outcomes.

The crisis-inclusive PIT Kolmogorov-Smirnov tests produced:

| Model | KS p-value |
|---|---:|
| Neural SDE | < 0.0001 |
| LSTM | < 0.0001 |

The results provide evidence against the null hypothesis of uniform PIT values for both predictive distributions over the analysed crisis-inclusive test period.

Therefore, neither model can be considered fully calibrated under this diagnostic over the complete evaluation period.

---

## Crisis and Regime Analysis

Additional analysis examines model behaviour during periods of substantial market stress.

The study defines two crisis/regime windows:

- **COVID-19 market crash:** February 2020 – April 2020
- **2022 bear market:** January 2022 – December 2022

These periods are analysed separately to investigate whether model performance and forecast uncertainty change across different market conditions.

---

## Repository Structure

```text
neural-sde-volatility-forecasting/
│
├── .gitignore
├── compare_results.py
├── deep_analysis.py
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── NSVM_Model_CRISIS.py
│   ├── LSTM_Benchmark_CRISIS.py
│   ├── compare_results.py
│   ├── forecast_plot.png
│   ├── pit_histogram.png
│   ├── predictive_distribution.png
│   └── training_curves.png
│
├── results/
│   ├── final_comparison_table.csv
│   ├── final_comparison_figure.png
│   ├── bootstrap_confidence_intervals.csv
│   ├── calibration_ks_test.csv
│   ├── regime_dm_tests.csv
│   ├── forecast_vs_actual.png
│   └── ...
│
└── data/
    └── [dataset excluded from public repository]
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kevingokani1/neural-sde-volatility-forecasting.git
cd neural-sde-volatility-forecasting
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib torch torchsde
```

---

## Running the Analysis

The principal comparison can be executed using:

```bash
python compare_results.py
```

Additional statistical and regime analysis is contained in:

```bash
python deep_analysis.py
```

Model implementations are contained within the `src/` directory.

---

## Research Outputs

The `results/` directory contains generated outputs from the empirical analysis, including:

- Model comparison tables
- Forecast visualisations
- Predictive distributions
- PIT diagnostics
- Calibration tests
- Bootstrap confidence intervals
- Regime-level statistical tests
- Training and forecasting outputs

---

## Limitations

Several limitations should be considered when interpreting the results.

First, the empirical analysis is based on a particular realised-volatility dataset and therefore the findings may not generalise to all financial assets or markets.

Second, the Gaussian observation specification imposes distributional assumptions that may not fully capture the heavy tails and asymmetry commonly observed in financial volatility.

Third, Neural SDE training involves stochastic simulation and numerical approximation, meaning computational choices such as the number of Monte Carlo paths and integration steps can influence the results.

Finally, the calibration diagnostics indicate that further work could investigate alternative predictive distributions, richer stochastic structures and additional calibration techniques.

---

## Academic Context

This project builds on research in Neural SDEs and continuous-time machine learning, including work on latent stochastic differential equation models.

A key reference underlying the Neural SDE methodology is:

> Kidger, P., Foster, J., Li, X. and Lyons, T. (2021). Neural SDEs as Infinite-Dimensional GANs. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.

---

## Author

**Kevin Gokani**

MSc Finance Data Science  
University of Birmingham

This repository contains work undertaken as part of an MSc dissertation in Finance Data Science.