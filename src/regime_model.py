"""
Phase 2 — Jump Model Regime Detection
======================================
Trains a Continuous Jump Model (CJM) on SPY features via expanding-window
walk-forward cross-validation to produce a fully out-of-sample bear-market
probability series p_bear.

Walk-forward protocol:
  - Minimum training: 5 years of daily data
  - Retrain once per year (not every month — stable enough, avoids overhead)
  - For each out-of-sample year: call predict_proba_online() so each daily
    forecast uses only data prior to that day (no look-ahead)
  - Backtest window: 2010-01-01 → end of available data

Outputs (saved to Data/processed/):
  p_bear.parquet      — daily bear probability, 2010 onward
  regime_labels.parquet — binary label (0=bull, 1=bear)
  regime_report.txt   — regime validation summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the repo root is on the path so jumpmodels can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import StandardScalerPD

OUT_DIR = ROOT / "Data" / "processed"

# ── Parameters ──────────────────────────────────────────────────────────────
JUMP_PENALTY   = 25.0   # λ: controls regime persistence (higher = fewer switches)
N_COMPONENTS   = 2      # bull / bear
CONT           = True   # Continuous JM → smooth bear probability
RANDOM_STATE   = 42
MIN_TRAIN_DAYS = 252 * 5  # 5 years minimum training window
BACKTEST_START = "2010-01-01"

# Known bear-market periods used for validation
BEAR_PERIODS = [
    ("2007-10-01", "2009-03-31", "GFC"),
    ("2020-02-15", "2020-03-31", "COVID"),
    ("2022-01-01", "2022-12-31", "Rate-hike"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_inputs() -> tuple[pd.DataFrame, pd.Series]:
    """Load SPY features and SPY returns (for fit sort_by='cumret')."""
    features = pd.read_parquet(OUT_DIR / "spy_features.parquet")
    # SPY is a macro file — load adj_close from raw and compute log-returns
    raw = ROOT / "Data" / "raw" / "Macro and Prices" / "SPY.csv"
    spy = pd.read_csv(raw, usecols=["m_date", "m_close_dividend_and_split_adjusted"],
                      parse_dates=["m_date"])
    spy = spy.rename(columns={"m_date": "date",
                               "m_close_dividend_and_split_adjusted": "adj_close"})
    spy = spy.dropna(subset=["adj_close"]).set_index("date").sort_index()
    returns = np.log(spy["adj_close"]).diff().rename("SPY").dropna()
    # Align on common index
    idx = features.index.intersection(returns.index)
    return features.loc[idx], returns.loc[idx]


def scale_features(X_train: pd.DataFrame,
                   X_test:  pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on train, apply to both train and test."""
    scaler = StandardScalerPD()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward engine
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_p_bear(features: pd.DataFrame,
                        spy_ret:  pd.Series) -> pd.DataFrame:
    """
    Expanding-window walk-forward.

    For each calendar year in the backtest window:
      1. Train CJM on all data up to Jan 1 of that year.
      2. Predict bear probability for that full year via predict_proba_online().

    Returns a DataFrame with columns [p_bull, p_bear] indexed by date.
    """
    backtest_dates = features.loc[BACKTEST_START:].index
    years = sorted(backtest_dates.year.unique())

    all_proba = []

    for year in years:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end   = pd.Timestamp(f"{year}-12-31")

        # --- Training set: everything strictly before this year ---
        X_train_raw = features.loc[:year_start - pd.Timedelta(days=1)]
        ret_train   = spy_ret.loc[:year_start - pd.Timedelta(days=1)]

        if len(X_train_raw) < MIN_TRAIN_DAYS:
            print(f"  [SKIP] {year}: only {len(X_train_raw)} training days "
                  f"(need {MIN_TRAIN_DAYS})")
            continue

        # --- Test set: this calendar year ---
        X_test_raw = features.loc[year_start:year_end]
        if len(X_test_raw) == 0:
            continue

        # Scale
        X_train_sc, X_test_sc = scale_features(X_train_raw, X_test_raw)
        ret_train_aligned = ret_train.reindex(X_train_sc.index)

        # Train CJM
        model = JumpModel(
            n_components=N_COMPONENTS,
            jump_penalty=JUMP_PENALTY,
            cont=CONT,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_sc, ret_ser=ret_train_aligned, sort_by="cumret")

        # Online prediction (no look-ahead)
        proba = model.predict_proba_online(X_test_sc)
        proba.columns = ["p_bull", "p_bear"]

        all_proba.append(proba)
        print(f"  [OK] {year}: trained on {len(X_train_raw)} days, "
              f"predicted {len(X_test_raw)} days  "
              f"| avg p_bear = {proba['p_bear'].mean():.3f}  "
              f"| in-sample states: {dict(model.labels_.value_counts().sort_index())}")

    result = pd.concat(all_proba).sort_index()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_regimes(p_bear: pd.Series) -> str:
    """
    Check that p_bear is elevated during known bear markets.
    Returns a formatted report string.
    """
    lines = [
        "=" * 60,
        "Regime Validation — Known Bear Markets",
        "=" * 60,
        f"{'Period':<12} {'Start':<12} {'End':<12} {'Avg p_bear':>10} {'Max p_bear':>10}",
        "-" * 60,
    ]
    for start, end, label in BEAR_PERIODS:
        window = p_bear.loc[start:end]
        if len(window) == 0:
            lines.append(f"{label:<12} {'N/A (outside backtest window)':>44}")
            continue
        avg = window.mean()
        mx  = window.max()
        flag = "  ✓" if avg > 0.50 else "  ✗ (LOW — check λ)"
        lines.append(f"{label:<12} {start:<12} {end:<12} {avg:>10.3f} {mx:>10.3f}{flag}")

    lines += [
        "",
        f"Full period avg p_bear : {p_bear.mean():.3f}  (expected ~0.2–0.35)",
        f"Days with p_bear > 0.5 : {(p_bear > 0.5).sum()} / {len(p_bear)}  "
        f"({(p_bear > 0.5).mean()*100:.1f}%)",
        "=" * 60,
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 2 — CJM Regime Detection  (λ={})".format(JUMP_PENALTY))
    print("=" * 60)

    features, spy_ret = load_inputs()
    print(f"[OK]  Features loaded : {features.shape}  "
          f"({features.index.min().date()} → {features.index.max().date()})")

    print(f"\nRunning walk-forward (backtest start: {BACKTEST_START})...")
    proba_df = walk_forward_p_bear(features, spy_ret)

    p_bear  = proba_df["p_bear"]
    labels  = (p_bear > 0.5).astype(int).rename("regime")  # 0=bull, 1=bear

    # Save
    proba_df.to_parquet(OUT_DIR / "p_bear.parquet")
    labels.to_frame().to_parquet(OUT_DIR / "regime_labels.parquet")
    print(f"\n[SAVED] p_bear.parquet        → {OUT_DIR}")
    print(f"[SAVED] regime_labels.parquet → {OUT_DIR}")

    # Validate
    report = validate_regimes(p_bear)
    print("\n" + report)
    (OUT_DIR / "regime_report.txt").write_text(report)
    print(f"[SAVED] regime_report.txt     → {OUT_DIR}")

    # Monthly summary (useful for portfolio construction)
    monthly_p_bear = p_bear.resample("ME").last().rename("p_bear_eom")
    monthly_p_bear.to_frame().to_parquet(OUT_DIR / "p_bear_monthly.parquet")
    print(f"[SAVED] p_bear_monthly.parquet → {OUT_DIR}")
    print(f"\nMonthly p_bear (last 12 months):\n{monthly_p_bear.tail(12).to_string()}")


if __name__ == "__main__":
    main()
