"""
Hyperparameter Tuning
======================
Grid search over:
  1. CJM jump penalty λ  — controls how quickly the model exits bear regimes
  2. EQUITY_MIN           — floor equity weight (limits defensiveness in bear)
  3. EQUITY_MAX           — ceiling equity weight (aggression in bull)
  4. bond_mode            — duration profile when p_bear is high

Methodology:
  IS  period: 2010-2019  (parameter selection)
  OOS period: 2020-2025  (honest out-of-sample evaluation)

The CJM p_bear series is re-computed for each λ using the same walk-forward
protocol (expanding window, no look-ahead). Portfolio params are then swept
over the cached p_bear series, which is cheap.

Best IS params are selected, then reported on OOS.
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR   = ROOT / "Data" / "processed"
TUNE_DIR  = ROOT / "Data" / "tuning"
TUNE_DIR.mkdir(parents=True, exist_ok=True)

IS_START  = "2010-01-01"
IS_END    = "2019-12-31"
OOS_START = "2020-01-01"
OOS_END   = "2025-12-31"
ANN       = 252

# ── Grids ────────────────────────────────────────────────────────────────────
LAMBDA_GRID      = [5, 10, 25, 50, 100]
EQUITY_MIN_GRID  = [0.20, 0.30, 0.40, 0.50]
EQUITY_MAX_GRID  = [0.80, 0.90]
# Bond allocation when p_bear >= 0.65 (full bear)
# "long"    : 50% EDV + 30% VGLT + 20% EMHY  (current — flight-to-quality)
# "medium"  : 30% VGLT + 40% VGIT + 30% VGSH (intermediate — balanced)
# "short"   : 20% VGSH + 40% SHV + 40% BIL   (cash-like — rising-rate safe)
BOND_MODE_GRID   = ["long", "medium", "short"]

SLEEVE_A_PCT = 0.60  # fixed (60% inv-vol / 40% momentum within equity)
MAX_POSITION = 0.08
MAX_EM       = 0.20
MAX_FI       = 0.90
VOL_TARGET   = 0.12
VOL_LOOKBACK = 60
MAX_SCALE    = 1.00


# ══════════════════════════════════════════════════════════════════════════════
# CJM re-training (one per λ value)
# ══════════════════════════════════════════════════════════════════════════════

def compute_p_bear_for_lambda(lam: float,
                               features: pd.DataFrame,
                               spy_ret:  pd.Series) -> pd.Series:
    """Run walk-forward CJM for a given jump penalty λ. Returns monthly p_bear."""
    from jumpmodels.jump import JumpModel
    from jumpmodels.preprocess import StandardScalerPD

    BACKTEST_START = "2010-01-01"
    MIN_TRAIN_DAYS = 252 * 5

    backtest_dates = features.loc[BACKTEST_START:].index
    years = sorted(backtest_dates.year.unique())
    all_proba = []

    for year in years:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end   = pd.Timestamp(f"{year}-12-31")

        X_train_raw = features.loc[:year_start - pd.Timedelta(days=1)]
        ret_train   = spy_ret.loc[:year_start - pd.Timedelta(days=1)]
        if len(X_train_raw) < MIN_TRAIN_DAYS:
            continue

        X_test_raw = features.loc[year_start:year_end]
        if len(X_test_raw) == 0:
            continue

        scaler       = StandardScalerPD()
        X_train_sc   = scaler.fit_transform(X_train_raw)
        X_test_sc    = scaler.transform(X_test_raw)
        ret_aligned  = ret_train.reindex(X_train_sc.index)

        model = JumpModel(n_components=2, jump_penalty=lam, cont=True,
                          random_state=42)
        model.fit(X_train_sc, ret_ser=ret_aligned, sort_by="cumret")

        proba = model.predict_proba_online(X_test_sc)
        proba.columns = ["p_bull", "p_bear"]
        all_proba.append(proba)

    daily = pd.concat(all_proba).sort_index()["p_bear"]
    monthly = daily.resample("ME").last().rename("p_bear_eom")
    return monthly


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio construction (fast — no CJM refit)
# ══════════════════════════════════════════════════════════════════════════════

TIER1_EQUITY = [
    "VGT","VHT","VFH","VCR","VDC","VIS","VAW","VDE","VPU","VOX","VNQ",
    "EWJ","EWG","EWU","EWL","EWQ","EWI","EWP","EWD","EWN","EWO","EWK",
    "EWS","EWA","EWC","EWT","EWH",
    "EWZ","EWW","EWY","EWM","EZA","THD","TUR","EPHE","GXC","ECH","EPU",
]
EM_EQUITY    = ["EWZ","EWW","EWY","EWM","EZA","THD","TUR","EPHE","GXC","ECH","EPU"]
FIXED_INCOME = ["EDV","VGLT","VGIT","VGSH","SHV","BIL"]
EM_CREDIT    = ["EMHY"]
ALL_ASSETS   = list(dict.fromkeys(TIER1_EQUITY + FIXED_INCOME + EM_CREDIT))


def _bond_weights(p_bear: float, bond_mode: str) -> pd.Series:
    """Duration ladder with three modes for the full-bear regime."""
    if p_bear < 0.20:
        alloc = {"SHV": 0.40, "BIL": 0.40, "VGSH": 0.20}
    elif p_bear < 0.35:
        alloc = {"VGIT": 0.40, "VGSH": 0.40, "SHV": 0.20}
    elif p_bear < 0.50:
        alloc = {"EDV": 0.25, "VGLT": 0.25, "VGIT": 0.25, "VGSH": 0.25}
    elif p_bear < 0.65:
        alloc = {"EDV": 0.40, "VGLT": 0.40, "VGIT": 0.20}
    else:
        if bond_mode == "long":
            alloc = {"EDV": 0.50, "VGLT": 0.30, "EMHY": 0.20}
        elif bond_mode == "medium":
            alloc = {"VGLT": 0.30, "VGIT": 0.40, "VGSH": 0.30}
        else:  # short
            alloc = {"VGSH": 0.20, "SHV": 0.40, "BIL": 0.40}
    return pd.Series(alloc)


def _apply_limits(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0)
    em_total = w[EM_EQUITY].sum()
    if em_total > MAX_EM:
        w[EM_EQUITY] *= MAX_EM / em_total

    eq_names = TIER1_EQUITY
    for _ in range(20):
        over = w[eq_names] > MAX_POSITION
        if not over.any():
            break
        excess = (w[eq_names][over] - MAX_POSITION).sum()
        w[eq_names] = w[eq_names].clip(upper=MAX_POSITION)
        under = w[eq_names][w[eq_names] < MAX_POSITION]
        if not under.empty:
            w[under.index] += excess * (under / under.sum())
        else:
            w["BIL"] = w.get("BIL", 0.0) + excess

    fi_names = FIXED_INCOME + EM_CREDIT
    fi_total = w[fi_names].sum()
    if fi_total > MAX_FI:
        scale   = MAX_FI / fi_total
        excess  = (1 - scale) * fi_total
        w[fi_names] *= scale
        w["BIL"] = w.get("BIL", 0.0) + excess
    return w


def _vol_scale(weights: pd.Series, cov_cache: dict,
               as_of: pd.Timestamp) -> float:
    """Use pre-computed covariance matrix (keyed by date)."""
    if as_of not in cov_cache:
        return 1.0
    cov_mx = cov_cache[as_of]
    tickers = list(cov_mx.index)
    w_full  = weights.reindex(tickers, fill_value=0.0).values
    var     = w_full @ cov_mx.values @ w_full * ANN
    return min(MAX_SCALE, VOL_TARGET / max(np.sqrt(var), 1e-4))


def precompute_cov_cache(returns: pd.DataFrame,
                          month_ends: pd.DatetimeIndex) -> dict:
    """
    Pre-compute EWMA covariance matrices at each month-end once.
    Returns {date: DataFrame (tickers × tickers)}.
    """
    tickers = [t for t in ALL_ASSETS if t in returns.columns]
    ret_sub = returns[tickers]
    cache   = {}
    for dt in month_ends:
        hist = ret_sub.loc[:dt].iloc[-VOL_LOOKBACK:]
        if len(hist) < 20:
            continue
        # EWMA covariance: last slice of the rolling result
        cov_full = hist.ewm(span=VOL_LOOKBACK // 2, min_periods=10).cov()
        cov_mx   = cov_full.iloc[-len(tickers):]
        cov_mx.index = pd.Index(tickers, name="ticker")
        cache[dt] = cov_mx
    return cache


def build_weights(p_bear_m: pd.Series, inv_vol_w: pd.DataFrame,
                  momentum_w: pd.DataFrame, cov_cache: dict,
                  equity_max: float, equity_min: float,
                  sleeve_a: float, bond_mode: str) -> pd.DataFrame:
    sleeve_b = 1.0 - sleeve_a
    dates = (p_bear_m.index
             .intersection(inv_vol_w.index)
             .intersection(momentum_w.index))
    rows = []
    for dt in dates:
        pb    = float(p_bear_m.loc[dt])
        eq_w  = equity_max - pb * (equity_max - equity_min)
        fi_w  = 1.0 - eq_w

        a_w   = inv_vol_w.loc[dt] * eq_w * sleeve_a
        b_w   = momentum_w.loc[dt] * eq_w * sleeve_b
        c_raw = _bond_weights(pb, bond_mode)
        c_w   = (c_raw * fi_w).reindex(ALL_ASSETS, fill_value=0.0)

        raw = pd.Series(0.0, index=ALL_ASSETS)
        for s in [a_w, b_w]:
            for t, v in s.items():
                if t in raw.index:
                    raw[t] += v
        for t, v in c_w.items():
            if t in raw.index:
                raw[t] += v

        raw   = _apply_limits(raw)
        scale = _vol_scale(raw, cov_cache, dt)
        scaled = raw * scale
        resid  = 1.0 - scaled.sum()
        if "BIL" in scaled.index:
            scaled["BIL"] += max(resid, 0.0)
        scaled.name = dt
        rows.append(scaled)

    df = pd.DataFrame(rows)
    df.index.name = "date"
    return df


def compute_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    dates  = sorted(weights.index)
    chunks = []
    for i, dt in enumerate(dates):
        w        = weights.loc[dt]
        next_dt  = dates[i+1] if i+1 < len(dates) else returns.index[-1]
        mask     = (returns.index > dt) & (returns.index <= next_dt)
        daily    = returns.loc[mask]
        if daily.empty:
            continue
        common   = [t for t in w.index if t in returns.columns and w[t] > 0]
        w_sub    = w[common]; w_sub /= w_sub.sum()
        chunks.append(daily[common].fillna(0.0).dot(w_sub))
    return pd.concat(chunks).sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def perf(r: pd.Series) -> dict:
    cagr   = r.mean() * ANN
    vol    = r.std() * np.sqrt(ANN)
    sharpe = cagr / vol if vol > 0 else np.nan
    cum    = (1 + r).cumprod()
    dd     = (cum / cum.cummax() - 1).min()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Hyperparameter Tuning  (IS: 2010-2019 | OOS: 2020-2025)")
    print("=" * 70)

    # Load static inputs
    returns    = pd.read_parquet(OUT_DIR / "returns.parquet")
    inv_vol_w  = pd.read_parquet(OUT_DIR / "inv_vol_weights.parquet")
    momentum_w = pd.read_parquet(OUT_DIR / "momentum_weights.parquet")
    bm_ret     = pd.read_parquet(OUT_DIR / "benchmark_returns.parquet")["AOA"]

    raw_csv = ROOT / "Data" / "raw" / "Macro and Prices" / "SPY.csv"
    spy_raw = pd.read_csv(raw_csv,
                          usecols=["m_date","m_close_dividend_and_split_adjusted"],
                          parse_dates=["m_date"])
    spy_raw = (spy_raw.rename(columns={"m_date":"date",
                                        "m_close_dividend_and_split_adjusted":"adj"})
                      .dropna().set_index("date").sort_index())
    spy_ret  = np.log(spy_raw["adj"]).diff().rename("SPY").dropna()
    features = pd.read_parquet(OUT_DIR / "spy_features.parquet")
    idx      = features.index.intersection(spy_ret.index)
    features, spy_ret = features.loc[idx], spy_ret.loc[idx]

    bm_is  = perf(bm_ret.loc[IS_START:IS_END])
    bm_oos = perf(bm_ret.loc[OOS_START:OOS_END])

    results = []

    # ── Step 1: compute p_bear for each λ ────────────────────────────────────
    p_bear_cache = {}
    for lam in LAMBDA_GRID:
        print(f"\n[λ={lam:>4.0f}] Training CJM walk-forward...", flush=True)
        p_bear_cache[lam] = compute_p_bear_for_lambda(lam, features, spy_ret)
        print(f"        avg p_bear = {p_bear_cache[lam].mean():.3f}  "
              f"n_months = {len(p_bear_cache[lam])}")

    # ── Pre-compute covariance matrices once (shared across all combos) ───────
    all_month_ends = (inv_vol_w.index
                      .union(momentum_w.index)
                      .sort_values())
    print("\nPre-computing EWMA covariance matrices...", flush=True)
    cov_cache = precompute_cov_cache(returns, all_month_ends)
    print(f"  Cached {len(cov_cache)} month-end covariance matrices.")

    # ── Step 2: portfolio grid search ─────────────────────────────────────────
    print(f"\nPortfolio grid search  "
          f"({len(EQUITY_MAX_GRID)}×{len(EQUITY_MIN_GRID)}×{len(BOND_MODE_GRID)} combos per λ)...")

    for lam, eq_max, eq_min, bmode in product(
            LAMBDA_GRID, EQUITY_MAX_GRID, EQUITY_MIN_GRID, BOND_MODE_GRID):

        if eq_min >= eq_max:
            continue

        p_bear_m = p_bear_cache[lam]
        w = build_weights(p_bear_m, inv_vol_w, momentum_w, cov_cache,
                          eq_max, eq_min, SLEEVE_A_PCT, bmode)
        r = compute_returns(w, returns)

        p_is  = perf(r.loc[IS_START:IS_END])
        p_oos = perf(r.loc[OOS_START:OOS_END])

        results.append(dict(
            lam=lam, eq_max=eq_max, eq_min=eq_min, bond_mode=bmode,
            is_sharpe=p_is["sharpe"],   is_cagr=p_is["cagr"],
            is_maxdd=p_is["maxdd"],     is_excess=p_is["cagr"] - bm_is["cagr"],
            oos_sharpe=p_oos["sharpe"], oos_cagr=p_oos["cagr"],
            oos_maxdd=p_oos["maxdd"],   oos_excess=p_oos["cagr"] - bm_oos["cagr"],
        ))

    res_df = pd.DataFrame(results)
    res_df.to_csv(TUNE_DIR / "grid_results.csv", index=False)

    # ── Step 3: select best IS Sharpe ────────────────────────────────────────
    best_idx = res_df["is_sharpe"].idxmax()
    best     = res_df.loc[best_idx]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION (highest IS Sharpe)")
    print("=" * 70)
    print(f"  λ            = {best.lam}")
    print(f"  equity_max   = {best.eq_max:.0%}")
    print(f"  equity_min   = {best.eq_min:.0%}")
    print(f"  bond_mode    = {best.bond_mode}")
    print()
    print(f"  IS  Sharpe={best.is_sharpe:.2f}  CAGR={best.is_cagr:.1%}  "
          f"MaxDD={best.is_maxdd:.1%}  Excess={best.is_excess:+.1%}")
    print(f"  OOS Sharpe={best.oos_sharpe:.2f}  CAGR={best.oos_cagr:.1%}  "
          f"MaxDD={best.oos_maxdd:.1%}  Excess={best.oos_excess:+.1%}")
    print()
    print(f"  Benchmark:")
    print(f"  IS  Sharpe={bm_is['sharpe']:.2f}  CAGR={bm_is['cagr']:.1%}  MaxDD={bm_is['maxdd']:.1%}")
    print(f"  OOS Sharpe={bm_oos['sharpe']:.2f}  CAGR={bm_oos['cagr']:.1%}  MaxDD={bm_oos['maxdd']:.1%}")

    # ── Step 4: top-10 IS configurations ─────────────────────────────────────
    print("\nTop 10 IS configurations by Sharpe:")
    top10 = res_df.nlargest(10, "is_sharpe")[
        ["lam","eq_max","eq_min","bond_mode",
         "is_sharpe","is_excess","oos_sharpe","oos_excess"]
    ].round(3)
    print(top10.to_string(index=False))

    # ── Step 5: save best p_bear and weights ─────────────────────────────────
    best_lam  = int(best.lam)
    best_pbm  = p_bear_cache[best_lam]
    best_w    = build_weights(best_pbm, inv_vol_w, momentum_w, cov_cache,
                               best.eq_max, best.eq_min, SLEEVE_A_PCT, best.bond_mode)
    best_r    = compute_returns(best_w, returns)

    best_pbm.to_frame().to_parquet(TUNE_DIR / "best_p_bear_monthly.parquet")
    best_w.to_parquet(TUNE_DIR / "best_portfolio_weights.parquet")
    best_r.to_frame("strategy").to_parquet(TUNE_DIR / "best_portfolio_returns.parquet")

    print(f"\n[SAVED] grid_results.csv, best_* files → {TUNE_DIR}")

    # ── Step 6: year-by-year for best config ─────────────────────────────────
    print("\nYear-by-year (best config vs AOA):")
    header = f"{'Year':<6} {'Strat':>7} {'AOA':>7} {'Excess':>7} {'p_bear':>7}"
    print(header); print("-" * 38)
    for yr in range(2010, 2026):
        s  = str(yr)
        r  = best_r.loc[s]
        b  = bm_ret.loc[s]
        pb = best_pbm[best_pbm.index.year == yr].mean()
        if len(r) > 0:
            print(f"{yr:<6} {r.sum():>7.1%} {b.sum():>7.1%} "
                  f"{r.sum()-b.sum():>7.1%} {pb:>7.3f}")


if __name__ == "__main__":
    main()
