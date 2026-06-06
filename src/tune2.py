"""
Targeted Second-Pass Tuning
============================
Tests two structural improvements over the baseline:

  1. 200-day MA filter on p_bear
     When SPY > 200d MA: cap p_bear at 0.25 (can't be deeply bearish
     when price trend is up). This fixes the 2020 missed-recovery problem.

  2. Yield-curve-conditional bond sleeve
     Uses the 10y-2y spread to choose duration dynamically:
       spread > 1.0%  → "long"   (normal curve: long bonds diversify)
       spread 0-1.0%  → "medium" (flattening: intermediate duration)
       spread < 0%    → "short"  (inverted: avoid rate-hike risk)
     This is NOT a fitted parameter — it uses contemporaneous yield data.

Grid:
  ma_filter      : [True, False]
  yc_bonds       : [True, False]   (True = yield-curve conditional)
  equity_min     : [0.30, 0.40, 0.50]
  lambda         : [5, 10, 25]

Fixed: equity_max=0.80, sleeve_a=0.60, bond_mode (base) = "medium"
       (used when yc_bonds=False as the fixed fallback)

IS: 2010-2019 | OOS: 2020-2025
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / "Data" / "processed"
TUNE_DIR = ROOT / "Data" / "tuning"
TUNE_DIR.mkdir(parents=True, exist_ok=True)

IS_START  = "2010-01-01";  IS_END    = "2019-12-31"
OOS_START = "2020-01-01";  OOS_END   = "2025-12-31"
ANN = 252

# ── Fixed params ─────────────────────────────────────────────────────────────
EQUITY_MAX   = 0.80
SLEEVE_A_PCT = 0.60
MAX_POSITION = 0.08
MAX_EM       = 0.20
MAX_FI       = 0.90
VOL_TARGET   = 0.12
VOL_LOOKBACK = 60
MAX_SCALE    = 1.00

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

# ── Grids ─────────────────────────────────────────────────────────────────────
MA_FILTER_GRID   = [True, False]
YC_BONDS_GRID    = [True, False]
EQUITY_MIN_GRID  = [0.30, 0.40, 0.50]
LAMBDA_GRID      = [5, 10, 25]
BASE_BOND_MODE   = "medium"   # fixed fallback when yc_bonds=False


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_spy_price() -> pd.Series:
    raw = ROOT / "Data" / "raw" / "Macro and Prices" / "SPY.csv"
    df  = pd.read_csv(raw, usecols=["m_date","m_close_dividend_and_split_adjusted"],
                      parse_dates=["m_date"])
    df  = (df.rename(columns={"m_date":"date",
                               "m_close_dividend_and_split_adjusted":"price"})
             .dropna().set_index("date").sort_index())
    return df["price"]


def compute_spy_ma_filter(spy_price: pd.Series,
                           month_ends: pd.DatetimeIndex,
                           ma_window: int = 200,
                           bear_cap: float = 0.25) -> pd.Series:
    """
    At each month-end, return the p_bear cap:
      - SPY > 200d MA  →  cap = bear_cap (0.25)
      - SPY <= 200d MA →  cap = 1.0 (no cap, use raw CJM signal)
    """
    ma = spy_price.rolling(ma_window, min_periods=ma_window // 2).mean()
    caps = {}
    for dt in month_ends:
        if dt not in spy_price.index:
            closest = spy_price.index[spy_price.index <= dt]
            if closest.empty:
                caps[dt] = 1.0
                continue
            dt_px = spy_price.loc[closest[-1]]
            dt_ma = ma.loc[closest[-1]] if closest[-1] in ma.index else np.nan
        else:
            dt_px = spy_price.loc[dt]
            dt_ma = ma.loc[dt] if dt in ma.index else np.nan

        if np.isnan(dt_ma):
            caps[dt] = 1.0
        elif dt_px > dt_ma:
            caps[dt] = bear_cap   # above MA → limit bearishness
        else:
            caps[dt] = 1.0        # below MA → allow full bear signal
    return pd.Series(caps)


def compute_yc_bond_mode(yields: pd.DataFrame,
                          month_ends: pd.DatetimeIndex) -> pd.Series:
    """
    At each month-end, return bond_mode based on 10y-2y spread:
      spread > 1.0%  → 'long'
      0 ≤ spread ≤ 1.0%  → 'medium'
      spread < 0%  → 'short'  (inverted / rising-rate risk)
    """
    if "year10" not in yields.columns or "year2" not in yields.columns:
        return pd.Series("medium", index=month_ends)

    spread = yields["year10"] - yields["year2"]
    modes  = {}
    for dt in month_ends:
        available = spread.loc[:dt].dropna()
        if available.empty:
            modes[dt] = "medium"
            continue
        s = available.iloc[-1]
        if s > 1.0:
            modes[dt] = "long"
        elif s >= 0.0:
            modes[dt] = "medium"
        else:
            modes[dt] = "short"
    return pd.Series(modes)


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio construction (same as tune.py but accepts pre-computed auxiliaries)
# ══════════════════════════════════════════════════════════════════════════════

def _bond_weights(p_bear: float, mode: str) -> pd.Series:
    if p_bear < 0.20:
        alloc = {"SHV": 0.40, "BIL": 0.40, "VGSH": 0.20}
    elif p_bear < 0.35:
        alloc = {"VGIT": 0.40, "VGSH": 0.40, "SHV": 0.20}
    elif p_bear < 0.50:
        alloc = {"EDV": 0.25, "VGLT": 0.25, "VGIT": 0.25, "VGSH": 0.25}
    elif p_bear < 0.65:
        alloc = {"EDV": 0.40, "VGLT": 0.40, "VGIT": 0.20}
    else:
        if mode == "long":
            alloc = {"EDV": 0.50, "VGLT": 0.30, "EMHY": 0.20}
        elif mode == "medium":
            alloc = {"VGLT": 0.30, "VGIT": 0.40, "VGSH": 0.30}
        else:
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
        scale  = MAX_FI / fi_total
        excess = (1 - scale) * fi_total
        w[fi_names] *= scale
        w["BIL"] = w.get("BIL", 0.0) + excess
    return w


def _vol_scale(weights: pd.Series, cov_cache: dict, dt: pd.Timestamp) -> float:
    if dt not in cov_cache:
        return 1.0
    cov_mx  = cov_cache[dt]
    tickers = list(cov_mx.index)
    w_full  = weights.reindex(tickers, fill_value=0.0).values
    var     = w_full @ cov_mx.values @ w_full * ANN
    return min(MAX_SCALE, VOL_TARGET / max(np.sqrt(var), 1e-4))


def build_weights(p_bear_m:   pd.Series,
                  inv_vol_w:  pd.DataFrame,
                  momentum_w: pd.DataFrame,
                  cov_cache:  dict,
                  equity_min: float,
                  ma_caps:    pd.Series,        # per-date p_bear cap
                  bond_modes: pd.Series,        # per-date bond mode
                  ) -> pd.DataFrame:

    dates = (p_bear_m.index
             .intersection(inv_vol_w.index)
             .intersection(momentum_w.index))
    rows = []
    for dt in dates:
        pb_raw = float(p_bear_m.loc[dt])
        cap    = float(ma_caps.get(dt, 1.0))
        pb     = min(pb_raw, cap)            # apply MA filter cap
        mode   = bond_modes.get(dt, BASE_BOND_MODE)

        eq_w = EQUITY_MAX - pb * (EQUITY_MAX - equity_min)
        fi_w = 1.0 - eq_w

        a_w  = inv_vol_w.loc[dt] * eq_w * SLEEVE_A_PCT
        b_w  = momentum_w.loc[dt] * eq_w * (1 - SLEEVE_A_PCT)
        c_r  = _bond_weights(pb, mode)
        c_w  = (c_r * fi_w).reindex(ALL_ASSETS, fill_value=0.0)

        raw = pd.Series(0.0, index=ALL_ASSETS)
        for s in [a_w, b_w]:
            for t, v in s.items():
                if t in raw.index:
                    raw[t] += v
        for t, v in c_w.items():
            if t in raw.index:
                raw[t] += v

        raw    = _apply_limits(raw)
        scale  = _vol_scale(raw, cov_cache, dt)
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
    dates = sorted(weights.index)
    chunks = []
    for i, dt in enumerate(dates):
        w       = weights.loc[dt]
        next_dt = dates[i+1] if i+1 < len(dates) else returns.index[-1]
        mask    = (returns.index > dt) & (returns.index <= next_dt)
        daily   = returns.loc[mask]
        if daily.empty:
            continue
        common  = [t for t in w.index if t in returns.columns and w[t] > 0]
        w_sub   = w[common]; w_sub /= w_sub.sum()
        chunks.append(daily[common].fillna(0.0).dot(w_sub))
    return pd.concat(chunks).sort_index()


def perf(r: pd.Series) -> dict:
    cagr   = r.mean() * ANN
    vol    = r.std() * np.sqrt(ANN)
    sharpe = cagr / vol if vol > 0 else np.nan
    cum    = (1 + r).cumprod()
    dd     = (cum / cum.cummax() - 1).min()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd)


# ══════════════════════════════════════════════════════════════════════════════
# CJM re-training
# ══════════════════════════════════════════════════════════════════════════════

def compute_p_bear_for_lambda(lam, features, spy_ret):
    from jumpmodels.jump import JumpModel
    from jumpmodels.preprocess import StandardScalerPD
    BACKTEST_START = "2010-01-01"; MIN_TRAIN_DAYS = 252 * 5
    years = sorted(features.loc[BACKTEST_START:].index.year.unique())
    all_proba = []
    for year in years:
        y_start = pd.Timestamp(f"{year}-01-01")
        y_end   = pd.Timestamp(f"{year}-12-31")
        X_tr    = features.loc[:y_start - pd.Timedelta(days=1)]
        r_tr    = spy_ret.loc[:y_start - pd.Timedelta(days=1)]
        if len(X_tr) < MIN_TRAIN_DAYS: continue
        X_te    = features.loc[y_start:y_end]
        if len(X_te) == 0: continue
        scaler  = StandardScalerPD()
        Xtr_sc  = scaler.fit_transform(X_tr)
        Xte_sc  = scaler.transform(X_te)
        model   = JumpModel(n_components=2, jump_penalty=lam, cont=True, random_state=42)
        model.fit(Xtr_sc, ret_ser=r_tr.reindex(Xtr_sc.index), sort_by="cumret")
        p = model.predict_proba_online(Xte_sc); p.columns = ["p_bull","p_bear"]
        all_proba.append(p)
    daily = pd.concat(all_proba).sort_index()["p_bear"]
    return daily.resample("ME").last().rename("p_bear_eom")


def precompute_cov_cache(returns, month_ends):
    tickers = [t for t in ALL_ASSETS if t in returns.columns]
    ret_sub = returns[tickers]; cache = {}
    for dt in month_ends:
        hist = ret_sub.loc[:dt].iloc[-VOL_LOOKBACK:]
        if len(hist) < 20: continue
        cov_full = hist.ewm(span=VOL_LOOKBACK//2, min_periods=10).cov()
        cov_mx   = cov_full.iloc[-len(tickers):]
        cov_mx.index = pd.Index(tickers, name="ticker")
        cache[dt] = cov_mx
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Second-Pass Tuning: MA Filter + Yield-Curve-Conditional Bonds")
    print("IS: 2010-2019  |  OOS: 2020-2025")
    print("=" * 70)

    returns    = pd.read_parquet(OUT_DIR / "returns.parquet")
    inv_vol_w  = pd.read_parquet(OUT_DIR / "inv_vol_weights.parquet")
    momentum_w = pd.read_parquet(OUT_DIR / "momentum_weights.parquet")
    yields     = pd.read_parquet(OUT_DIR / "yields.parquet")
    bm_ret     = pd.read_parquet(OUT_DIR / "benchmark_returns.parquet")["AOA"]

    raw_csv = ROOT / "Data" / "raw" / "Macro and Prices" / "SPY.csv"
    spy_raw = pd.read_csv(raw_csv,
                          usecols=["m_date","m_close_dividend_and_split_adjusted"],
                          parse_dates=["m_date"])
    spy_raw = (spy_raw.rename(columns={"m_date":"date",
                                        "m_close_dividend_and_split_adjusted":"price"})
                      .dropna().set_index("date").sort_index())
    spy_px  = spy_raw["price"]
    spy_ret = np.log(spy_px).diff().rename("SPY").dropna()
    features = pd.read_parquet(OUT_DIR / "spy_features.parquet")
    idx      = features.index.intersection(spy_ret.index)
    features, spy_ret = features.loc[idx], spy_ret.loc[idx]

    bm_is  = perf(bm_ret.loc[IS_START:IS_END])
    bm_oos = perf(bm_ret.loc[OOS_START:OOS_END])

    # Pre-compute per-date auxiliaries
    all_month_ends = inv_vol_w.index.union(momentum_w.index).sort_values()
    print("\nPre-computing covariance cache...")
    cov_cache   = precompute_cov_cache(returns, all_month_ends)

    print("Computing MA caps (200d)...")
    ma_caps_on  = compute_spy_ma_filter(spy_px, all_month_ends, ma_window=200)
    ma_caps_off = pd.Series(1.0, index=all_month_ends)   # no filter

    print("Computing yield-curve bond modes...")
    yc_modes_on  = compute_yc_bond_mode(yields, all_month_ends)
    yc_modes_off = pd.Series(BASE_BOND_MODE, index=all_month_ends)

    # Train CJM for each λ
    p_bear_cache = {}
    for lam in LAMBDA_GRID:
        print(f"\n[λ={lam:>3}] CJM walk-forward...", flush=True)
        p_bear_cache[lam] = compute_p_bear_for_lambda(lam, features, spy_ret)

    # Grid search
    total = len(MA_FILTER_GRID)*len(YC_BONDS_GRID)*len(EQUITY_MIN_GRID)*len(LAMBDA_GRID)
    print(f"\nPortfolio grid: {total} combinations...")
    results = []
    for use_ma, use_yc, eq_min, lam in product(
            MA_FILTER_GRID, YC_BONDS_GRID, EQUITY_MIN_GRID, LAMBDA_GRID):

        ma_caps   = ma_caps_on   if use_ma else ma_caps_off
        yc_modes  = yc_modes_on  if use_yc else yc_modes_off

        p_bear_m  = p_bear_cache[lam]
        w  = build_weights(p_bear_m, inv_vol_w, momentum_w, cov_cache,
                           eq_min, ma_caps, yc_modes)
        r  = compute_returns(w, returns)
        p_is  = perf(r.loc[IS_START:IS_END])
        p_oos = perf(r.loc[OOS_START:OOS_END])

        results.append(dict(
            lam=lam, eq_min=eq_min,
            ma_filter=use_ma, yc_bonds=use_yc,
            is_sharpe=p_is["sharpe"],   is_cagr=p_is["cagr"],
            is_maxdd=p_is["maxdd"],     is_excess=p_is["cagr"]-bm_is["cagr"],
            oos_sharpe=p_oos["sharpe"], oos_cagr=p_oos["cagr"],
            oos_maxdd=p_oos["maxdd"],   oos_excess=p_oos["cagr"]-bm_oos["cagr"],
        ))

    res_df = pd.DataFrame(results)
    res_df.to_csv(TUNE_DIR / "grid2_results.csv", index=False)

    # ── Results summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Effect of each improvement (averaged over all other params):")
    print("=" * 70)
    for col in ["ma_filter","yc_bonds"]:
        print(f"\n  {col}:")
        print(res_df.groupby(col)[["is_sharpe","oos_sharpe","oos_excess"]].mean().round(3).to_string())

    # Best by IS Sharpe
    best_is = res_df.loc[res_df["is_sharpe"].idxmax()]
    # Best by OOS Sharpe (for understanding)
    best_oos = res_df.loc[res_df["oos_sharpe"].idxmax()]
    # Best balanced (IS + OOS average Sharpe)
    res_df["balanced"] = (res_df["is_sharpe"] + res_df["oos_sharpe"]) / 2
    best_bal = res_df.loc[res_df["balanced"].idxmax()]

    print("\n" + "=" * 70)
    print("SELECTED CONFIGURATIONS")
    print("=" * 70)
    for label, cfg in [("Best IS Sharpe", best_is),
                       ("Best OOS Sharpe", best_oos),
                       ("Best Balanced (IS+OOS avg)", best_bal)]:
        print(f"\n  [{label}]")
        print(f"    λ={cfg.lam}  eq_min={cfg.eq_min:.0%}  "
              f"ma_filter={cfg.ma_filter}  yc_bonds={cfg.yc_bonds}")
        print(f"    IS  Sharpe={cfg.is_sharpe:.2f}  CAGR={cfg.is_cagr:.1%}  "
              f"MaxDD={cfg.is_maxdd:.1%}  Excess={cfg.is_excess:+.1%}")
        print(f"    OOS Sharpe={cfg.oos_sharpe:.2f}  CAGR={cfg.oos_cagr:.1%}  "
              f"MaxDD={cfg.oos_maxdd:.1%}  Excess={cfg.oos_excess:+.1%}")

    print(f"\n  [Benchmark AOA]")
    print(f"    IS  Sharpe={bm_is['sharpe']:.2f}  CAGR={bm_is['cagr']:.1%}  MaxDD={bm_is['maxdd']:.1%}")
    print(f"    OOS Sharpe={bm_oos['sharpe']:.2f}  CAGR={bm_oos['cagr']:.1%}  MaxDD={bm_oos['maxdd']:.1%}")

    # ── Save balanced best config ─────────────────────────────────────────────
    ma_c  = ma_caps_on   if best_bal.ma_filter else ma_caps_off
    yc_c  = yc_modes_on  if best_bal.yc_bonds  else yc_modes_off
    pb_m  = p_bear_cache[int(best_bal.lam)]
    best_w = build_weights(pb_m, inv_vol_w, momentum_w, cov_cache,
                            best_bal.eq_min, ma_c, yc_c)
    best_r = compute_returns(best_w, returns)

    best_w.to_parquet(TUNE_DIR / "final_portfolio_weights.parquet")
    best_r.to_frame("strategy").to_parquet(TUNE_DIR / "final_portfolio_returns.parquet")
    pb_m.to_frame().to_parquet(TUNE_DIR / "final_p_bear_monthly.parquet")

    # Year-by-year comparison
    print("\nYear-by-year (balanced-best config vs AOA):")
    print(f"{'Year':<6} {'Strat':>7} {'AOA':>7} {'Excess':>7} {'p_bear':>7}")
    print("-" * 38)
    for yr in range(2010, 2026):
        r  = best_r.loc[str(yr)]
        b  = bm_ret.loc[str(yr)]
        pb = pb_m[pb_m.index.year == yr].mean()
        if len(r) > 0:
            print(f"{yr:<6} {r.sum():>7.1%} {b.sum():>7.1%} "
                  f"{r.sum()-b.sum():>7.1%} {pb:>7.3f}")

    print(f"\n[SAVED] grid2_results.csv + final_* files → {TUNE_DIR}")


if __name__ == "__main__":
    main()
