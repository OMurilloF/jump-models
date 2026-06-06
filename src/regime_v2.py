"""
V2 Layer 1 — Regime Engine
===========================
Produces three families of regime signals, all via walk-forward CJM (no
look-ahead, expanding window, yearly retrain, same protocol as v1):

  1. Per-asset bull probability  p_bull_assets  (38 equity ETFs)
     Each ETF's regime is derived from ITS OWN return features — fixes the
     signal/portfolio mismatch flagged in the post-mortem.

  2. Global equity regime  p_bear_global
     SPY features augmented with 2 orthogonal macro features (VIX level,
     2s10s slope). Governs the equity/bond split.

  3. Rates regime  p_bear_rates
     CJM on a synthetic 10y-duration bond return (from Treasury yields, full
     history back to 1990). p_bear_rates high = rising-yield / bond-bear
     regime → avoid duration (this is the data-driven replacement for the
     hand-set yield-curve thresholds).

Design choices to avoid overfitting:
  - Same jump_penalty (lambda=25) used everywhere; no per-asset tuning.
  - Macro augmentation limited to 2 features.
  - Walk-forward identical to v1 (5y min train, yearly retrain, online predict).

Outputs (Data/processed/v2/):
  p_bull_assets.parquet         daily, columns = 38 equity tickers
  p_bull_assets_monthly.parquet month-end
  p_bear_global.parquet         daily
  p_bear_global_monthly.parquet month-end
  p_bear_rates.parquet          daily
  p_bear_rates_monthly.parquet  month-end
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import StandardScalerPD

PROC    = ROOT / "Data" / "processed"
RAW     = ROOT / "Data" / "raw" / "Macro and Prices"
OUT     = PROC / "v2"
OUT.mkdir(parents=True, exist_ok=True)

JUMP_PENALTY   = 25.0
N_INIT         = 5            # slightly reduced for 38-asset run; CJM is stable
MIN_TRAIN_DAYS = 252 * 5
BACKTEST_START = "2010-01-01"
RANDOM_STATE   = 42

TIER1_EQUITY = [
    "VGT","VHT","VFH","VCR","VDC","VIS","VAW","VDE","VPU","VOX","VNQ",
    "EWJ","EWG","EWU","EWL","EWQ","EWI","EWP","EWD","EWN","EWO","EWK",
    "EWS","EWA","EWC","EWT","EWH",
    "EWZ","EWW","EWY","EWM","EZA","THD","TUR","EPHE","GXC","ECH","EPU",
]


# ── Feature engineering (shared) ─────────────────────────────────────────────

def _ewm_stats(ret: pd.Series, span: int) -> pd.DataFrame:
    ewm_ret = ret.ewm(span=span, min_periods=span // 2).mean() * 252
    ewm_dd  = (ret.clip(upper=0).pow(2)
               .ewm(span=span, min_periods=span // 2).mean().pow(0.5)) * np.sqrt(252)
    sortino = ewm_ret / ewm_dd.replace(0, np.nan)
    return pd.DataFrame({
        f"ewm_ret_{span}": ewm_ret,
        f"ewm_dd_{span}":  ewm_dd,
        f"sortino_{span}": sortino,
    })


def build_return_features(ret: pd.Series) -> pd.DataFrame:
    """9 EWM features (ret/dd/sortino x 21/63/126) from a return series."""
    feats = pd.concat([_ewm_stats(ret, s) for s in (21, 63, 126)], axis=1)
    return feats.dropna()


# ── Walk-forward CJM (returns daily p_bear) ───────────────────────────────────

def walk_forward_cjm(features: pd.DataFrame,
                     ret_for_sort: pd.Series,
                     label: str = "") -> pd.Series:
    """Expanding-window walk-forward CJM. Returns daily p_bear series."""
    years = sorted(features.loc[BACKTEST_START:].index.year.unique())
    proba_chunks = []
    for year in years:
        y0 = pd.Timestamp(f"{year}-01-01")
        y1 = pd.Timestamp(f"{year}-12-31")
        X_tr = features.loc[:y0 - pd.Timedelta(days=1)]
        if len(X_tr) < MIN_TRAIN_DAYS:
            continue
        X_te = features.loc[y0:y1]
        if len(X_te) == 0:
            continue
        scaler = StandardScalerPD()
        Xtr = scaler.fit_transform(X_tr)
        Xte = scaler.transform(X_te)
        r_tr = ret_for_sort.reindex(Xtr.index)
        model = JumpModel(n_components=2, jump_penalty=JUMP_PENALTY, cont=True,
                          random_state=RANDOM_STATE, n_init=N_INIT)
        model.fit(Xtr, ret_ser=r_tr, sort_by="cumret")
        p = model.predict_proba_online(Xte)
        p.columns = ["p_bull", "p_bear"]
        proba_chunks.append(p["p_bear"])
    if not proba_chunks:
        return pd.Series(dtype=float)
    return pd.concat(proba_chunks).sort_index().rename(label)


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_spy() -> pd.Series:
    df = pd.read_csv(RAW / "SPY.csv",
                     usecols=["m_date", "m_close_dividend_and_split_adjusted"],
                     parse_dates=["m_date"])
    df = (df.rename(columns={"m_date": "date",
                              "m_close_dividend_and_split_adjusted": "px"})
            .dropna().set_index("date").sort_index())
    return df["px"]


def load_vix() -> pd.Series:
    df = pd.read_csv(RAW / "VIX_History.csv", parse_dates=["DATE"])
    return df.set_index("DATE").sort_index()["CLOSE"].rename("vix")


def load_yields() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Treasuries_Historical Data.csv",
                     parse_dates=["date"], dayfirst=True)
    df.columns = [c.strip() for c in df.columns]
    return df.set_index("date").sort_index()


# ── Build each regime family ──────────────────────────────────────────────────

def build_global_regime(returns: pd.DataFrame) -> pd.Series:
    """SPY features + VIX level + 2s10s slope → p_bear_global."""
    spy_px  = load_spy()
    spy_ret = np.log(spy_px).diff().dropna()
    feats   = build_return_features(spy_ret)

    vix     = load_vix()
    yields  = load_yields()
    slope   = (yields["year10"] - yields["year2"]).rename("slope2s10s")

    # Macro features aligned to SPY feature dates, forward-filled
    macro = pd.DataFrame(index=feats.index)
    macro["vix"]   = vix.reindex(feats.index, method="ffill")
    macro["slope"] = slope.reindex(feats.index, method="ffill")

    X = feats.join(macro).dropna()
    print(f"  [global] feature matrix {X.shape}")
    return walk_forward_cjm(X, spy_ret.reindex(X.index), "p_bear_global")


def build_rates_regime() -> pd.Series:
    """Synthetic 10y bond return from yields → p_bear_rates (rising-yield bear)."""
    yields = load_yields()
    y10    = yields["year10"].dropna()
    # Approx price return of a 10y-duration bond: r = -D * dY (yields in %)
    bond_ret = (-10.0 * y10.diff() / 100.0).dropna().rename("bond_ret")
    feats = build_return_features(bond_ret)
    print(f"  [rates]  feature matrix {feats.shape}")
    return walk_forward_cjm(feats, bond_ret.reindex(feats.index), "p_bear_rates")


def build_asset_regimes(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-asset p_bull (= 1 - p_bear) for each Tier-1 equity ETF."""
    cols = {}
    for i, tkr in enumerate(TIER1_EQUITY, 1):
        if tkr not in returns.columns:
            print(f"  [{i:2d}/38] {tkr}: MISSING — skipped")
            continue
        ret   = returns[tkr].dropna()
        feats = build_return_features(ret)
        if len(feats.loc[BACKTEST_START:]) == 0:
            print(f"  [{i:2d}/38] {tkr}: no backtest data — skipped")
            continue
        p_bear = walk_forward_cjm(feats, ret.reindex(feats.index), tkr)
        cols[tkr] = (1.0 - p_bear).rename(tkr)        # store p_bull
        n = len(p_bear)
        avg = p_bear.mean()
        print(f"  [{i:2d}/38] {tkr}: {n} days, avg p_bear={avg:.3f}", flush=True)
    return pd.DataFrame(cols).sort_index()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("V2 Layer 1 — Regime Engine")
    print("=" * 64)

    returns = pd.read_parquet(PROC / "returns.parquet")

    print("\n[1/3] Global equity regime (SPY + macro)...")
    p_global = build_global_regime(returns)
    p_global.to_frame().to_parquet(OUT / "p_bear_global.parquet")
    p_global.resample("ME").last().to_frame().to_parquet(OUT / "p_bear_global_monthly.parquet")
    print(f"  saved p_bear_global  (avg={p_global.mean():.3f}, n={len(p_global)})")

    print("\n[2/3] Rates regime (synthetic 10y bond)...")
    p_rates = build_rates_regime()
    p_rates.to_frame().to_parquet(OUT / "p_bear_rates.parquet")
    p_rates.resample("ME").last().to_frame().to_parquet(OUT / "p_bear_rates_monthly.parquet")
    print(f"  saved p_bear_rates   (avg={p_rates.mean():.3f}, n={len(p_rates)})")

    print("\n[3/3] Per-asset equity regimes (38 ETFs)...")
    p_assets = build_asset_regimes(returns)
    p_assets.to_parquet(OUT / "p_bull_assets.parquet")
    p_assets.resample("ME").last().to_parquet(OUT / "p_bull_assets_monthly.parquet")
    print(f"\n  saved p_bull_assets  shape={p_assets.shape}")

    print("\nDone. All regime signals saved to", OUT)


if __name__ == "__main__":
    main()
