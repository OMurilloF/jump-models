"""
Phase 1 — Data Preparation
===========================
Loads all raw CSVs from Data/raw/Macro and Prices/, standardizes them into
clean DataFrames, and produces:

  1. prices.parquet       — daily adj-close prices, all tickers
  2. returns.parquet      — daily log-returns, all tickers
  3. spy_features.parquet — 9-feature matrix for CJM regime model
  4. vix.parquet          — VIX close series
  5. yields.parquet       — Treasury yield curve (13 tenors)
  6. oas.parquet          — ICE OAS spread series

All outputs saved to Data/processed/.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Data" / "raw" / "Macro and Prices"
OUT_DIR = ROOT / "Data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe ───────────────────────────────────────────────────────────────────
# All equity / fixed-income ETFs (excluding benchmark, macro, and special files)
MACRO_FILES = {"SPY.csv", "VIX_History.csv", "Treasuries_Historical Data.csv",
               "^IDCOTSTR.csv", ".gitkeep"}
BENCHMARK   = "AOA.csv"

ETF_TICKERS = sorted([
    f.stem for f in RAW_DIR.glob("*.csv")
    if f.name not in MACRO_FILES and f.name != BENCHMARK
])

ALL_TICKERS = ETF_TICKERS + ["AOA"]

# Columns present in the wide ETF CSVs
DATE_COL   = "m_date"
ADJ_CLOSE  = "m_close_dividend_and_split_adjusted"
LOG_RET    = "c_log_returns_dividend_and_split_adjusted"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ETF price & return loader
# ══════════════════════════════════════════════════════════════════════════════

def load_etf(ticker: str) -> pd.DataFrame:
    """Load a single ETF CSV; return DataFrame with [adj_close, log_ret]."""
    path = RAW_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, usecols=[DATE_COL, ADJ_CLOSE, LOG_RET],
                     parse_dates=[DATE_COL])
    df = df.rename(columns={DATE_COL: "date",
                             ADJ_CLOSE: "adj_close",
                             LOG_RET: "log_ret"})
    df = df.dropna(subset=["adj_close"])
    df = df.set_index("date").sort_index()
    # Some rows have adj_close but missing log_ret (first row of each ticker)
    # Recompute from adj_close to fill any gaps
    df["log_ret"] = np.log(df["adj_close"]).diff()
    return df[["adj_close", "log_ret"]]


def build_price_return_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build aligned price and return panels across all tickers.
    Returns (prices_df, returns_df) — both indexed by date, columns = tickers.
    """
    prices, returns = {}, {}
    missing = []
    for ticker in ALL_TICKERS:
        path = RAW_DIR / f"{ticker}.csv"
        if not path.exists():
            missing.append(ticker)
            continue
        df = load_etf(ticker)
        prices[ticker]  = df["adj_close"]
        returns[ticker] = df["log_ret"]

    if missing:
        print(f"[WARN] Missing files for: {missing}")

    prices_df  = pd.DataFrame(prices)
    returns_df = pd.DataFrame(returns)

    # Forward-fill price panel by at most 5 days (e.g. thin EM markets)
    # but leave returns NaN where there was no real trade
    prices_df = prices_df.ffill(limit=5)

    print(f"[OK]  Price panel   : {prices_df.shape}  "
          f"({prices_df.index.min().date()} → {prices_df.index.max().date()})")
    print(f"[OK]  Return panel  : {returns_df.shape}")

    # Coverage report
    coverage = returns_df.notna().mean().sort_values()
    thin = coverage[coverage < 0.80]
    if not thin.empty:
        print("\n[WARN] Tickers with <80% data coverage (consider excluding):")
        print(thin.to_string())

    return prices_df, returns_df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SPY feature matrix for CJM
# ══════════════════════════════════════════════════════════════════════════════

def load_etf_raw(ticker: str) -> pd.DataFrame:
    """Load any ticker CSV by name (including macro files like SPY)."""
    return load_etf(ticker)

def _ewm_stats(ret: pd.Series, span: int) -> pd.DataFrame:
    """Compute EWM return, downside deviation, and Sortino for a given span."""
    ewm_ret  = ret.ewm(span=span, min_periods=span // 2).mean() * 252
    ewm_dd   = ret.clip(upper=0).pow(2).ewm(span=span, min_periods=span // 2).mean().pow(0.5) * np.sqrt(252)
    sortino  = ewm_ret / ewm_dd.replace(0, np.nan)
    return pd.DataFrame({
        f"ewm_ret_{span}":  ewm_ret,
        f"ewm_dd_{span}":   ewm_dd,
        f"sortino_{span}":  sortino,
    })


def build_spy_features(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 9-feature matrix used as CJM inputs, derived from SPY returns.

    Features (3 metrics × 3 horizons):
      - EWM annualized return      : span = 21, 63, 126
      - EWM downside deviation     : span = 21, 63, 126
      - EWM Sortino ratio          : span = 21, 63, 126
    """
    # SPY is a macro file loaded separately from the ETF panel
    spy_df  = load_etf_raw("SPY")
    spy_ret = spy_df["log_ret"].dropna()

    features = pd.concat(
        [_ewm_stats(spy_ret, span) for span in [21, 63, 126]],
        axis=1
    )
    features = features.dropna()

    print(f"[OK]  SPY features  : {features.shape}  "
          f"({features.index.min().date()} → {features.index.max().date()})")
    print(f"      Columns: {list(features.columns)}")
    return features


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VIX loader
# ══════════════════════════════════════════════════════════════════════════════

def load_vix() -> pd.Series:
    """Load VIX history; return daily close series."""
    path = RAW_DIR / "VIX_History.csv"
    df = pd.read_csv(path, parse_dates=["DATE"])
    df = df.rename(columns={"DATE": "date", "CLOSE": "vix"})
    df = df.set_index("date").sort_index()
    ser = df["vix"]
    print(f"[OK]  VIX series    : {len(ser)} rows  "
          f"({ser.index.min().date()} → {ser.index.max().date()})")
    return ser


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Treasury yield curve loader
# ══════════════════════════════════════════════════════════════════════════════

def load_yields() -> pd.DataFrame:
    """Load treasury yield curve; returns DataFrame with 13 tenor columns."""
    path = RAW_DIR / "Treasuries_Historical Data.csv"
    # Dates in DD/MM/YYYY format
    df = pd.read_csv(path, parse_dates=["date"], dayfirst=True)
    df = df.set_index("date").sort_index()
    # Rename for clarity
    df.columns = [c.strip() for c in df.columns]
    print(f"[OK]  Yields        : {df.shape}  "
          f"({df.index.min().date()} → {df.index.max().date()})")
    print(f"      Tenors: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ICE OAS spread loader
# ══════════════════════════════════════════════════════════════════════════════

def load_oas() -> pd.Series:
    """Load ICE OAS index (^IDCOTSTR); return daily close series."""
    path = RAW_DIR / "^IDCOTSTR.csv"
    df = pd.read_csv(path, usecols=[DATE_COL, ADJ_CLOSE],
                     parse_dates=[DATE_COL])
    df = df.rename(columns={DATE_COL: "date", ADJ_CLOSE: "oas"})
    df = df.dropna(subset=["oas"])
    df = df.set_index("date").sort_index()
    ser = df["oas"]
    print(f"[OK]  OAS series    : {len(ser)} rows  "
          f"({ser.index.min().date()} → {ser.index.max().date()})")
    return ser


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 1 — Data Preparation")
    print("=" * 60)

    # --- ETF panels ---
    prices_df, returns_df = build_price_return_panels()
    prices_df.to_parquet(OUT_DIR / "prices.parquet")
    returns_df.to_parquet(OUT_DIR / "returns.parquet")
    print(f"[SAVED] prices.parquet  → {OUT_DIR / 'prices.parquet'}")
    print(f"[SAVED] returns.parquet → {OUT_DIR / 'returns.parquet'}")

    # --- SPY features ---
    spy_features = build_spy_features(returns_df)
    spy_features.to_parquet(OUT_DIR / "spy_features.parquet")
    print(f"[SAVED] spy_features.parquet → {OUT_DIR / 'spy_features.parquet'}")

    # --- Macro ---
    vix = load_vix()
    vix.to_frame("vix").to_parquet(OUT_DIR / "vix.parquet")
    print(f"[SAVED] vix.parquet → {OUT_DIR / 'vix.parquet'}")

    yields = load_yields()
    yields.to_parquet(OUT_DIR / "yields.parquet")
    print(f"[SAVED] yields.parquet → {OUT_DIR / 'yields.parquet'}")

    oas = load_oas()
    oas.to_frame("oas").to_parquet(OUT_DIR / "oas.parquet")
    print(f"[SAVED] oas.parquet → {OUT_DIR / 'oas.parquet'}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    common_start = returns_df.dropna(how="all").index.min()
    common_end   = returns_df.dropna(how="all").index.max()
    backtest_ret = returns_df.loc["2010-01-01":]
    print(f"Full history       : {common_start.date()} → {common_end.date()}")
    print(f"Backtest window    : 2010-01-01 → {common_end.date()}")
    print(f"Tickers loaded     : {returns_df.shape[1]}")
    print(f"Tickers ≥2010 data : {backtest_ret.notna().any().sum()}")
    print("\nAll outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
