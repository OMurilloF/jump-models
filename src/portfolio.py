"""
Phase 4 — Portfolio Construction
==================================
Blends the three sleeves into final monthly portfolio weights, applies
position limits, and applies volatility targeting.

Architecture recap:
  Sleeve A (60% of equity) — inverse-vol core, 38 Tier-1 equity ETFs
  Sleeve B (40% of equity) — top-quintile momentum, 38 Tier-1 equity ETFs
  Sleeve C (all of bond)   — duration ladder, steered by p_bear

Regime overlay (p_bear → equity/bond split):
  equity_weight = EQUITY_MAX - p_bear * (EQUITY_MAX - EQUITY_MIN)
  bond_weight   = 1 - equity_weight   (always fully invested via BIL/SHV if needed)

Duration ladder (Sleeve C):
  p_bear < 0.20  →  short: 40% SHV + 40% BIL + 20% VGSH
  p_bear < 0.35  →  short-mid: 40% VGIT + 40% VGSH + 20% SHV
  p_bear < 0.50  →  barbell: 25% each EDV/VGLT/VGIT/VGSH
  p_bear < 0.65  →  long: 40% EDV + 40% VGLT + 20% VGIT
  p_bear >= 0.65 →  max-long: 50% EDV + 30% VGLT + 20% EMHY

Vol targeting:
  Scale weights so expected portfolio vol ≈ VOL_TARGET (12% ann.)
  Scale capped at 1.0 (no leverage in long-only mandate).
  Any residual cash from scaling goes to BIL.

Outputs saved to Data/processed/:
  portfolio_weights.parquet — monthly weights per ticker (rebalance dates)
  portfolio_returns.parquet — daily strategy returns (hold between rebalances)
  benchmark_returns.parquet — daily AOA returns (aligned to same window)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "Data" / "processed"

# ── Universe ─────────────────────────────────────────────────────────────────
US_SECTORS = ["VGT", "VHT", "VFH", "VCR", "VDC", "VIS", "VAW", "VDE",
               "VPU", "VOX", "VNQ"]
DM_INTL    = ["EWJ", "EWG", "EWU", "EWL", "EWQ", "EWI", "EWP", "EWD",
               "EWN", "EWO", "EWK", "EWS", "EWA", "EWC", "EWT", "EWH"]
EM_EQUITY  = ["EWZ", "EWW", "EWY", "EWM", "EZA", "THD", "TUR",
               "EPHE", "GXC", "ECH", "EPU"]
TIER1_EQUITY = US_SECTORS + DM_INTL + EM_EQUITY
FIXED_INCOME = ["EDV", "VGLT", "VGIT", "VGSH", "SHV", "BIL"]
EM_CREDIT    = ["EMHY"]
ALL_ASSETS   = TIER1_EQUITY + FIXED_INCOME + EM_CREDIT  # 45 names + AOA

# ── Parameters ───────────────────────────────────────────────────────────────
EQUITY_MAX   = 0.80   # max equity weight (full bull)
EQUITY_MIN   = 0.10   # min equity weight (full bear)
SLEEVE_A_PCT = 0.60   # Sleeve A share of equity allocation
SLEEVE_B_PCT = 0.40   # Sleeve B share of equity allocation

MAX_POSITION = 0.08   # single equity-name cap
MAX_EM       = 0.20   # total EM equity cap
MAX_FI       = 0.90   # total fixed income cap (regime can call for ~90% bonds)

VOL_TARGET   = 0.12   # 12% annualized
MAX_SCALE    = 1.00   # no leverage (long-only)
VOL_LOOKBACK = 60     # days for EWMA vol estimation

BACKTEST_START = "2010-01-01"


# ══════════════════════════════════════════════════════════════════════════════
# Duration ladder (Sleeve C)
# ══════════════════════════════════════════════════════════════════════════════

def bond_sleeve_weights(p_bear: float) -> pd.Series:
    """Map bear probability to duration-ladder weights (sum to 1.0)."""
    if p_bear < 0.20:
        alloc = {"SHV": 0.40, "BIL": 0.40, "VGSH": 0.20}
    elif p_bear < 0.35:
        alloc = {"VGIT": 0.40, "VGSH": 0.40, "SHV": 0.20}
    elif p_bear < 0.50:
        alloc = {"EDV": 0.25, "VGLT": 0.25, "VGIT": 0.25, "VGSH": 0.25}
    elif p_bear < 0.65:
        alloc = {"EDV": 0.40, "VGLT": 0.40, "VGIT": 0.20}
    else:
        alloc = {"EDV": 0.50, "VGLT": 0.30, "EMHY": 0.20}
    return pd.Series(alloc)


# ══════════════════════════════════════════════════════════════════════════════
# Position-limit enforcement
# ══════════════════════════════════════════════════════════════════════════════

def apply_position_limits(w: pd.Series) -> pd.Series:
    """
    Enforce position limits with iterative redistribution.

    Rules:
    - Equity names: hard cap at MAX_POSITION (8%) per name
    - EM equity combined: cap at MAX_EM (20%)
    - Fixed income combined (incl. EMHY): cap at MAX_FI (50%)
    - Fixed income individual names (Treasuries): no per-name cap
    - BIL / SHV: no cap (serve as cash parking)

    Excess from equity caps is redistributed to uncapped equity names
    iteratively until stable, then any remaining excess goes to BIL.
    """
    w = w.clip(lower=0.0)

    # ── EM equity combined cap ─────────────────────────────────────────────
    em_total = w[EM_EQUITY].sum()
    if em_total > MAX_EM:
        w[EM_EQUITY] *= MAX_EM / em_total

    # ── Per-name equity cap (iterative redistribution) ────────────────────
    equity_names = TIER1_EQUITY
    for _ in range(20):  # converges in a few iterations
        over = w[equity_names] > MAX_POSITION
        if not over.any():
            break
        excess = (w[equity_names][over] - MAX_POSITION).sum()
        w[equity_names] = w[equity_names].clip(upper=MAX_POSITION)
        # Redistribute excess to equity names not at their cap
        under = ~(w[equity_names] >= MAX_POSITION)
        if under.any():
            under_names = w[equity_names][under].index
            w[under_names] += excess * (w[under_names] / w[under_names].sum())
        else:
            # All equity names at cap — park excess in BIL
            w["BIL"] = w.get("BIL", 0.0) + excess

    # ── Fixed income combined cap ─────────────────────────────────────────
    fi_names = FIXED_INCOME + EM_CREDIT
    fi_total = w[fi_names].sum()
    if fi_total > MAX_FI:
        scale = MAX_FI / fi_total
        excess = (1 - scale) * fi_total
        w[fi_names] *= scale
        # Park excess in BIL (already in fi_names, so add to it directly)
        # This keeps weights summing to 1 via the vol-targeting step
        w["BIL"] = w.get("BIL", 0.0) + excess

    return w


# ══════════════════════════════════════════════════════════════════════════════
# Vol targeting
# ══════════════════════════════════════════════════════════════════════════════

def vol_scale(weights: pd.Series,
              returns: pd.DataFrame,
              as_of: pd.Timestamp) -> float:
    """
    Estimate current portfolio vol using recent return history and EWMA cov.
    Return the scale factor to hit VOL_TARGET (capped at MAX_SCALE).
    """
    # Use tickers that are in both weights and returns
    tickers = [t for t in weights.index if t in returns.columns and weights[t] > 0]
    if not tickers:
        return 1.0

    w = weights[tickers].values
    hist = returns[tickers].loc[:as_of].iloc[-VOL_LOOKBACK:]
    if len(hist) < 20:
        return 1.0

    # EWMA covariance
    cov = hist.ewm(span=VOL_LOOKBACK // 2, min_periods=10).cov().iloc[-len(tickers):]
    cov_mx = cov.values  # (n_tickers, n_tickers)

    port_var = w @ cov_mx @ w * 252
    port_vol = np.sqrt(max(port_var, 1e-8))

    scale = min(MAX_SCALE, VOL_TARGET / port_vol)
    return scale


# ══════════════════════════════════════════════════════════════════════════════
# Monthly weight builder
# ══════════════════════════════════════════════════════════════════════════════

def build_monthly_weights(p_bear_monthly: pd.Series,
                          inv_vol_w:      pd.DataFrame,
                          momentum_w:     pd.DataFrame,
                          returns:        pd.DataFrame) -> pd.DataFrame:
    """
    For each rebalance date, build the final portfolio weight vector.

    Signal timing: use p_bear from end of month T to set weights for month T+1
    (i.e., weights are determined on the last day of the month and applied
    starting the first day of the next month).
    """
    all_assets = list(dict.fromkeys(TIER1_EQUITY + FIXED_INCOME + EM_CREDIT))
    rows = []

    # Align rebalance dates — must exist in all three signal frames
    common_dates = (
        p_bear_monthly.index
        .intersection(inv_vol_w.index)
        .intersection(momentum_w.index)
    )

    for dt in common_dates:
        pb = float(p_bear_monthly.loc[dt])

        # ── Sleeve allocation split ───────────────────────────────────────────
        eq_w  = EQUITY_MAX - pb * (EQUITY_MAX - EQUITY_MIN)
        fi_w  = 1.0 - eq_w

        # ── Sleeve A: inv-vol core (60% of equity) ────────────────────────────
        a_total = eq_w * SLEEVE_A_PCT
        a_weights = inv_vol_w.loc[dt] * a_total   # already normalised within equity

        # ── Sleeve B: momentum top-quintile (40% of equity) ───────────────────
        b_total = eq_w * SLEEVE_B_PCT
        b_weights = momentum_w.loc[dt] * b_total  # already normalised within sleeve

        # ── Sleeve C: duration ladder (100% of bond allocation) ───────────────
        c_alloc   = bond_sleeve_weights(pb)
        c_weights = (c_alloc * fi_w).reindex(all_assets, fill_value=0.0)

        # ── Combine ───────────────────────────────────────────────────────────
        raw = pd.Series(0.0, index=all_assets)
        for series in [a_weights, b_weights]:
            for ticker, wt in series.items():
                if ticker in raw.index:
                    raw[ticker] += wt
        for ticker, wt in c_weights.items():
            if ticker in raw.index:
                raw[ticker] += wt

        # ── Position limits ───────────────────────────────────────────────────
        raw = apply_position_limits(raw)

        # ── Vol targeting ─────────────────────────────────────────────────────
        scale = vol_scale(raw, returns, dt)
        scaled = raw * scale
        cash_residual = 1.0 - scaled.sum()
        # Park residual cash in BIL
        if "BIL" in scaled.index:
            scaled["BIL"] = scaled.get("BIL", 0.0) + max(cash_residual, 0.0)

        scaled.name = dt
        rows.append(scaled)

    weights_df = pd.DataFrame(rows)
    weights_df.index.name = "date"
    return weights_df


# ══════════════════════════════════════════════════════════════════════════════
# Daily return calculation
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_returns(weights: pd.DataFrame,
                          returns: pd.DataFrame) -> pd.Series:
    """
    Apply monthly weights to daily returns. Weights are held constant within
    each month (rebalanced at month-end, applied starting next day).

    Returns a daily portfolio return series.
    """
    # Shift weights by 1 month: weights set on month-end T apply to month T+1
    # We do this by forward-filling the weight from each month-end date
    # onto the daily return dates of the following month.

    port_ret = []

    dates = sorted(weights.index)
    for i, rebal_date in enumerate(dates):
        w = weights.loc[rebal_date]

        # Return dates: from day after rebalance to the next rebalance date
        next_rebal = dates[i + 1] if i + 1 < len(dates) else returns.index[-1]
        mask = (returns.index > rebal_date) & (returns.index <= next_rebal)
        daily = returns.loc[mask]

        if daily.empty:
            continue

        # Keep only tickers present in returns
        common = [t for t in w.index if t in returns.columns and w[t] > 0]
        w_sub  = w[common]
        if w_sub.sum() == 0:
            continue
        w_sub = w_sub / w_sub.sum()  # renormalise after subsetting

        r_sub  = daily[common].fillna(0.0)
        p_ret  = r_sub.dot(w_sub)
        p_ret.name = "strategy"
        port_ret.append(p_ret)

    return pd.concat(port_ret).sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 4 — Portfolio Construction")
    print("=" * 60)

    # Load inputs
    returns      = pd.read_parquet(OUT_DIR / "returns.parquet")
    inv_vol_w    = pd.read_parquet(OUT_DIR / "inv_vol_weights.parquet")
    momentum_w   = pd.read_parquet(OUT_DIR / "momentum_weights.parquet")
    p_bear_m     = pd.read_parquet(OUT_DIR / "p_bear_monthly.parquet")["p_bear_eom"]

    # AOA benchmark returns
    aoa_ret = returns["AOA"].dropna().rename("AOA")

    print(f"[OK]  Returns loaded  : {returns.shape}")
    print(f"[OK]  p_bear monthly  : {len(p_bear_m)} months")
    print(f"[OK]  Inv-vol weights : {inv_vol_w.shape}")
    print(f"[OK]  Momentum weights: {momentum_w.shape}")

    # Build monthly weights
    print("\nBuilding monthly portfolio weights...")
    weights = build_monthly_weights(p_bear_m, inv_vol_w, momentum_w, returns)
    print(f"[OK]  Weight matrix   : {weights.shape}  "
          f"({weights.index.min().date()} → {weights.index.max().date()})")

    # Sanity checks
    w_sums = weights.sum(axis=1)
    print(f"      Weight sums     : min={w_sums.min():.4f}  max={w_sums.max():.4f}  "
          f"mean={w_sums.mean():.4f}  (should be ≈1.0)")
    # BIL/SHV are cash proxies — report max equity name separately
    equity_max = weights[TIER1_EQUITY].max().max()
    print(f"      Max equity pos  : {equity_max:.4f}  (limit={MAX_POSITION})")
    print(f"      Max BIL (cash)  : {weights['BIL'].max():.4f}")

    # Compute daily returns
    print("\nComputing daily portfolio returns...")
    port_ret = compute_daily_returns(weights, returns)
    print(f"[OK]  Daily returns   : {len(port_ret)} days  "
          f"({port_ret.index.min().date()} → {port_ret.index.max().date()})")

    # Align benchmark
    bm_ret = aoa_ret.reindex(port_ret.index).fillna(0.0)

    # Save
    weights.to_parquet(OUT_DIR / "portfolio_weights.parquet")
    port_ret.to_frame().to_parquet(OUT_DIR / "portfolio_returns.parquet")
    bm_ret.to_frame().to_parquet(OUT_DIR / "benchmark_returns.parquet")
    print(f"\n[SAVED] portfolio_weights.parquet  → {OUT_DIR}")
    print(f"[SAVED] portfolio_returns.parquet  → {OUT_DIR}")
    print(f"[SAVED] benchmark_returns.parquet  → {OUT_DIR}")

    # Quick performance preview
    print("\n--- Quick Performance Preview (net of 5bp transaction cost) ---")
    # Rough transaction cost: 5bp × monthly turnover
    turnover = weights.diff().abs().sum(axis=1).mean()
    tc_drag  = turnover * 0.0005 * 12  # annualised
    ann_factor = 252

    for label, r in [("Strategy", port_ret), ("AOA", bm_ret)]:
        cagr    = r.mean() * ann_factor
        vol     = r.std() * np.sqrt(ann_factor)
        sharpe  = cagr / vol if vol > 0 else np.nan
        cumret  = (1 + r).prod() - 1
        # Max drawdown
        cum_px  = (1 + r).cumprod()
        drawdown = (cum_px / cum_px.cummax() - 1).min()
        print(f"  {label:<10}: CAGR={cagr:.1%}  Vol={vol:.1%}  "
              f"Sharpe={sharpe:.2f}  MaxDD={drawdown:.1%}  CumRet={cumret:.0%}")

    print(f"\n  Est. annual TC drag (strategy): {tc_drag:.2%}")
    print(f"  Strategy CAGR net of TC       : {port_ret.mean()*ann_factor - tc_drag:.1%}")


if __name__ == "__main__":
    main()
