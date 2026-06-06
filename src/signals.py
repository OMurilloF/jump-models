"""
Phase 3 — Signal Construction
==============================
Builds the two equity signals used in the long-only strategy:

  Sleeve A — Inverse-Volatility (Smart Beta Core)
    For each month-end, weight each Tier-1 equity ETF by 1/vol(60d).
    Weights are normalised to sum to 1 within the equity universe.

  Sleeve B — Cross-Sectional Momentum (Long-Only Top Quintile)
    12-1 momentum = cumulative log-return from t-252 to t-21 (skip last month).
    Rank all Tier-1 equity ETFs. Top quintile (~8 names) receives equal weight;
    all others receive zero weight within this sleeve.

Both signals are computed at every month-end (last business day) and saved as
monthly weight DataFrames indexed by rebalance date.

Outputs saved to Data/processed/:
  inv_vol_weights.parquet    — Sleeve A monthly weights (equity universe)
  momentum_ranks.parquet     — raw momentum scores and ranks (all equity)
  momentum_weights.parquet   — Sleeve B monthly weights (top-quintile only)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "Data" / "processed"

# ── Universe definition ──────────────────────────────────────────────────────
US_SECTORS = ["VGT", "VHT", "VFH", "VCR", "VDC", "VIS", "VAW", "VDE",
               "VPU", "VOX", "VNQ"]

DM_INTL = ["EWJ", "EWG", "EWU", "EWL", "EWQ", "EWI", "EWP", "EWD",
            "EWN", "EWO", "EWK", "EWS", "EWA", "EWC", "EWT", "EWH"]

EM_EQUITY = ["EWZ", "EWW", "EWY", "EWM", "EZA", "THD", "TUR",
             "EPHE", "GXC", "ECH", "EPU"]

TIER1_EQUITY = US_SECTORS + DM_INTL + EM_EQUITY   # 38 names

FIXED_INCOME = ["EDV", "VGLT", "VGIT", "VGSH", "SHV", "BIL"]
EM_CREDIT    = ["EMHY"]

# ── Signal parameters ─────────────────────────────────────────────────────────
VOL_LOOKBACK  = 60    # days for realised vol (Sleeve A)
MOM_LONG      = 252   # lookback for momentum (12 months)
MOM_SKIP      = 21    # skip last month (avoids short-term reversal)
TOP_QUINTILE  = 0.20  # top 20% → long in Sleeve B
BACKTEST_START = "2010-01-01"

# ══════════════════════════════════════════════════════════════════════════════
# Sleeve A — Inverse-Volatility Weights
# ══════════════════════════════════════════════════════════════════════════════

def compute_inv_vol_weights(returns: pd.DataFrame,
                            universe: list[str],
                            month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """
    At each month-end, compute inverse-vol weights for the equity universe.

    vol(i, t) = annualised realised std of daily log-returns over past VOL_LOOKBACK days.
    weight(i) = (1 / vol(i)) / sum(1 / vol(j) for j in universe)

    Tickers with insufficient history are excluded (weight = 0).
    """
    ret = returns[universe]
    weights_rows = []

    for dt in month_ends:
        window = ret.loc[:dt].iloc[-VOL_LOOKBACK:]
        if len(window) < VOL_LOOKBACK // 2:
            continue

        vol = window.std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)

        inv_vol = (1.0 / vol).dropna()
        if inv_vol.empty:
            continue

        w = inv_vol / inv_vol.sum()
        w = w.reindex(universe).fillna(0.0)
        w.name = dt
        weights_rows.append(w)

    df = pd.DataFrame(weights_rows)
    df.index.name = "date"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Sleeve B — Cross-Sectional Momentum Weights
# ══════════════════════════════════════════════════════════════════════════════

def compute_momentum_signal(prices: pd.DataFrame,
                            universe: list[str],
                            month_ends: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    At each month-end, rank tickers by 12-1 momentum and assign weights.

    momentum(i, t) = cumulative log-return from (t - MOM_LONG) to (t - MOM_SKIP)
    Rank in descending order (highest momentum = rank 1).
    Top quintile receives equal weight; all others = 0.

    Returns:
        ranks_df   — DataFrame of momentum score and rank per ticker
        weights_df — DataFrame of Sleeve B weights
    """
    px = prices[universe]
    log_px = np.log(px)

    ranks_rows   = []
    weights_rows = []

    for dt in month_ends:
        # Locate price index positions
        hist = log_px.loc[:dt]
        if len(hist) < MOM_LONG + 5:
            continue

        # Price MOM_LONG days ago (start) and MOM_SKIP days ago (end, skip last month)
        p_end   = hist.iloc[-(MOM_SKIP + 1)]    # ~1 month ago
        p_start = hist.iloc[-(MOM_LONG + 1)]    # ~12 months ago

        mom_score = p_end - p_start             # cumulative log return 12-1
        valid     = mom_score.dropna()

        if len(valid) < 4:
            continue

        # Rank: 1 = best momentum
        rank = valid.rank(ascending=False, method="average")
        n    = len(valid)

        # Top quintile threshold
        cutoff = int(np.ceil(n * TOP_QUINTILE))
        top_names = rank[rank <= cutoff].index.tolist()

        # Equal weight within top quintile, 0 elsewhere
        w = pd.Series(0.0, index=universe)
        if top_names:
            w[top_names] = 1.0 / len(top_names)

        # Store rank row
        rank_row = pd.DataFrame({
            "score": mom_score,
            "rank":  rank.reindex(mom_score.index),
            "in_top_quintile": rank.reindex(mom_score.index) <= cutoff,
        })
        rank_row["date"] = dt
        ranks_rows.append(rank_row.reset_index().rename(columns={"index": "ticker"}))

        w.name = dt
        weights_rows.append(w)

    ranks_df   = pd.concat(ranks_rows, ignore_index=True).set_index(["date", "ticker"])
    weights_df = pd.DataFrame(weights_rows)
    weights_df.index.name = "date"
    return ranks_df, weights_df


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def print_signal_diagnostics(inv_vol_w: pd.DataFrame,
                             mom_w:     pd.DataFrame,
                             mom_ranks: pd.DataFrame) -> None:
    print("\n--- Sleeve A: Inverse-Vol Weight Diagnostics ---")
    avg_w = inv_vol_w.mean()
    print(f"  Avg weight range: {avg_w.min():.3f} – {avg_w.max():.3f}")
    print("  Top 5 avg-weight tickers:")
    print(avg_w.sort_values(ascending=False).head(5).to_string(float_format="{:.3f}".format))

    print("\n--- Sleeve B: Momentum Weight Diagnostics ---")
    # Average number of tickers selected per month
    n_selected = (mom_w > 0).sum(axis=1)
    print(f"  Avg tickers in top quintile per month: {n_selected.mean():.1f}  "
          f"(range: {n_selected.min()} – {n_selected.max()})")

    # Most frequently appearing in top quintile
    freq = (mom_w > 0).mean().sort_values(ascending=False)
    print("  Most frequent top-quintile tickers:")
    print(freq.head(8).to_string(float_format="{:.2%}".format))

    # Turnover: fraction of top-quintile names changing each month
    changes = (mom_w > 0).astype(int).diff().abs().sum(axis=1) / 2
    avg_portfolio_size = n_selected.mean()
    turnover = (changes / avg_portfolio_size).mean()
    print(f"  Avg monthly one-way turnover (momentum sleeve): {turnover:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 3 — Signal Construction")
    print("=" * 60)

    prices  = pd.read_parquet(OUT_DIR / "prices.parquet")
    returns = pd.read_parquet(OUT_DIR / "returns.parquet")

    # Month-end dates within backtest window
    all_month_ends = prices.loc[BACKTEST_START:].resample("ME").last().index
    # Only keep dates where we actually have price data
    valid_month_ends = pd.DatetimeIndex([
        dt for dt in all_month_ends if dt in prices.index
    ])
    print(f"[OK]  Rebalance dates: {len(valid_month_ends)}  "
          f"({valid_month_ends[0].date()} → {valid_month_ends[-1].date()})")

    # ── Sleeve A ──────────────────────────────────────────────────────────────
    print("\nComputing Sleeve A: Inverse-vol weights...")
    inv_vol_w = compute_inv_vol_weights(returns, TIER1_EQUITY, valid_month_ends)
    inv_vol_w.to_parquet(OUT_DIR / "inv_vol_weights.parquet")
    print(f"[SAVED] inv_vol_weights.parquet  shape: {inv_vol_w.shape}")

    # ── Sleeve B ──────────────────────────────────────────────────────────────
    print("\nComputing Sleeve B: Momentum ranks and weights...")
    mom_ranks, mom_w = compute_momentum_signal(prices, TIER1_EQUITY, valid_month_ends)
    mom_ranks.to_parquet(OUT_DIR / "momentum_ranks.parquet")
    mom_w.to_parquet(OUT_DIR / "momentum_weights.parquet")
    print(f"[SAVED] momentum_ranks.parquet   shape: {mom_ranks.shape}")
    print(f"[SAVED] momentum_weights.parquet shape: {mom_w.shape}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print_signal_diagnostics(inv_vol_w, mom_w, mom_ranks)

    print("\n" + "=" * 60)
    print(f"All signals saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
