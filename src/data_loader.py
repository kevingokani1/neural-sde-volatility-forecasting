# ============================================================
# data_loader.py -- CALENDAR-SPLIT VERSION
#
# Fixes the crisis-window gap: the previous 70/15/15 PERCENTAGE
# split happened to put COVID-19 (Feb-Apr 2020) and the 2022
# bear market entirely inside the training set. This version
# uses fixed CALENDAR cutoffs instead, so both crisis periods
# fall inside the TEST set -- a genuine out-of-sample stress
# test, not an in-sample one.
#
# Train : 2009-09-30 -- 2017-12-31  (~8 years)
# Val   : 2018-01-01 -- 2019-12-31  (2 years, pre-COVID, calm)
# Test  : 2020-01-01 -- 2026-07-31  (~6.5 years; includes
#         COVID crash, 2022 bear market, and recent data)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

CSV_PATH = DATA_DIR / "realized_variance_futures.csv"
TICKER = "ES"
RV_COLUMN = "rv5"

TRAIN_END = "2017-12-31"
VAL_END = "2019-12-31"


def load_rv_series(csv_path: Path = CSV_PATH, ticker: str = TICKER, rv_column: str = RV_COLUMN):
    """Returns (log_rv, rv, T, dates) for one ticker, chronologically sorted."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}.")

    df = pd.read_csv(csv_path, usecols=["date", "symbol", rv_column])
    df_ticker = df[df["symbol"] == ticker].copy()

    if len(df_ticker) == 0:
        available = df["symbol"].unique()
        raise ValueError(f"Ticker '{ticker}' not found. Available: {available[:15]}")

    df_ticker["date"] = pd.to_datetime(df_ticker["date"], dayfirst=True)
    df_ticker = df_ticker.sort_values("date").reset_index(drop=True)

    rv = df_ticker[rv_column].values.astype(np.float64)
    rv = np.clip(rv, 1e-12, None)
    log_rv = np.log(rv)
    T = len(log_rv)
    dates = pd.DatetimeIndex(df_ticker["date"])

    print(f"[data_loader] Loaded {T} trading days for {ticker} "
          f"({dates[0].date()} to {dates[-1].date()}), column='{rv_column}'")

    return log_rv, rv, T, dates


def calendar_split_indices(dates: pd.DatetimeIndex, train_end=TRAIN_END, val_end=VAL_END):
    """
    Returns (train_end_idx, val_end_idx) -- integer positions, NOT dates --
    so the rest of the pipeline (which works in integer array positions)
    doesn't need to change at all, only how these two numbers are computed.
    """
    train_end_idx = int(np.searchsorted(dates, pd.Timestamp(train_end), side="right"))
    val_end_idx = int(np.searchsorted(dates, pd.Timestamp(val_end), side="right"))
    return train_end_idx, val_end_idx


if __name__ == "__main__":
    log_rv, rv, T, dates = load_rv_series()
    train_end_idx, val_end_idx = calendar_split_indices(dates)
    print(f"\nTrain: {train_end_idx} days ({dates[0].date()} to {dates[train_end_idx-1].date()})")
    print(f"Val:   {val_end_idx - train_end_idx} days ({dates[train_end_idx].date()} to {dates[val_end_idx-1].date()})")
    print(f"Test:  {T - val_end_idx} days ({dates[val_end_idx].date()} to {dates[T-1].date()})")

    covid_mask = (dates >= "2020-02-15") & (dates <= "2020-04-30")
    bear_mask = (dates >= "2022-01-01") & (dates <= "2022-12-31")
    covid_in_test = int(np.sum(covid_mask[val_end_idx:]))
    bear_in_test = int(np.sum(bear_mask[val_end_idx:]))
    print(f"\nCOVID crash days in TEST set: {covid_in_test}")
    print(f"2022 bear market days in TEST set: {bear_in_test}")