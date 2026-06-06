"""
V2 Strategy — Layers 2-4
=========================
Combines the V2 regime engine (Layer 1, see regime_v2.py) with:

  Layer 2 — Multi-factor cross-sectional alpha (per equity ETF, monthly):
    * 12-1 momentum   (canonical)
    * 3-1  momentum   (faster trend)
    * trailing 252d Sharpe (quality / risk-adjusted persistence)
    Each factor is cross-sectionally z-scored, then EQUAL-WEIGHTED averaged.
    Equal weighting is a deliberate anti-overfitting choice (no fitted
    factor weights).

  Layer 3 — Ledoit-Wolf max-Sharpe optimizer (long-only):
    expected return  mu_i = alpha_z_i * p_bull_i   (regime-gated alpha)
    covariance       Ledoit-Wolf shrinkage (parameter-free)
    objective        maximise  w'mu / sqrt(w'Sigma w)
    constraints      0 <= w_i <= 8%, sum(w)=equity_budget, EM <= 20%
    Replaces inverse-vol + equal-weight-momentum heuristics.

  Layer 4 — Risk overlays (rule-based, industry-standard thresholds):
    * Equity/bond split from global regime
    * Recovery trigger: SPY 20d > +8% AND VIX < 22 AND SPY > 200d MA
      => cap global p_bear at 0.30 (catches V-shaped recoveries, e.g. 2020)
    * Bond duration governed by the rates regime (data-driven, no hand-set
      yield thresholds)

Vol targeting to 12% ann., scale capped at 1.0 (no leverage), residual->BIL.

Outputs (Data/processed/v2/):
  v2_weights.parquet
  v2_returns.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

PROC = ROOT / "Data" / "processed"
RAW  = ROOT / "Data" / "raw" / "Macro and Prices"
V2   = PROC / "v2"

ANN = 252

# ── Universe ─────────────────────────────────────────────────────────────────
US_SECTORS = ["VGT","VHT","VFH","VCR","VDC","VIS","VAW","VDE","VPU","VOX","VNQ"]
DM_INTL    = ["EWJ","EWG","EWU","EWL","EWQ","EWI","EWP","EWD","EWN","EWO","EWK",
              "EWS","EWA","EWC","EWT","EWH"]
EM_EQUITY  = ["EWZ","EWW","EWY","EWM","EZA","THD","TUR","EPHE","GXC","ECH","EPU"]
TIER1_EQUITY = US_SECTORS + DM_INTL + EM_EQUITY
BONDS        = ["EDV","VGLT","VGIT","VGSH","SHV","BIL"]
ALL_ASSETS   = TIER1_EQUITY + BONDS

# ── Parameters (deliberately few, none fitted on test data) ──────────────────
EQUITY_MAX   = 0.80
EQUITY_MIN   = 0.50      # from v1 tuning; robust across IS/OOS
MAX_POSITION = 0.08
MAX_EM       = 0.20
VOL_TARGET   = 0.12
VOL_LOOKBACK = 60
COV_LOOKBACK = 252       # 1y window for Ledoit-Wolf optimizer covariance
MOM_LONG     = 252
MOM_MID      = 63
MOM_SKIP     = 21
QUAL_WIN     = 252
BACKTEST_START = "2010-01-01"

# Recovery-trigger thresholds (industry-standard, NOT tuned on this data)
REC_RET_20D  = 0.08      # +8% 20-day SPY rally
REC_VIX_MAX  = 22.0      # VIX below 22 = non-stressed
REC_PBEAR_CAP = 0.30     # cap p_bear when recovery confirmed


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Multi-factor alpha
# ══════════════════════════════════════════════════════════════════════════════

def compute_alpha_scores(prices: pd.DataFrame, returns: pd.DataFrame,
                         month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Cross-sectional z-scored, equal-weighted multi-factor alpha per month."""
    log_px = np.log(prices[TIER1_EQUITY])
    ret    = returns[TIER1_EQUITY]
    rows = []
    for dt in month_ends:
        hist_px = log_px.loc[:dt]
        hist_r  = ret.loc[:dt]
        if len(hist_px) < MOM_LONG + 5:
            continue
        p_skip = hist_px.iloc[-(MOM_SKIP + 1)]
        p_long = hist_px.iloc[-(MOM_LONG + 1)]
        p_mid  = hist_px.iloc[-(MOM_MID + 1)]

        mom_12_1 = p_skip - p_long
        mom_3_1  = p_skip - p_mid
        qwin     = hist_r.iloc[-QUAL_WIN:]
        quality  = (qwin.mean() * ANN) / (qwin.std() * np.sqrt(ANN))

        def zscore(s):
            s = s.replace([np.inf, -np.inf], np.nan)
            mu, sd = s.mean(), s.std()
            return (s - mu) / sd if sd > 0 else s * 0.0

        alpha = (zscore(mom_12_1) + zscore(mom_3_1) + zscore(quality)) / 3.0
        alpha.name = dt
        rows.append(alpha)
    df = pd.DataFrame(rows)
    df.index.name = "date"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Ledoit-Wolf max-Sharpe optimizer
# ══════════════════════════════════════════════════════════════════════════════

def max_sharpe_weights(mu: pd.Series, cov: np.ndarray, tickers: list,
                       equity_budget: float) -> pd.Series:
    """
    Long-only max-Sharpe: maximise w'mu / sqrt(w'Sigma w)
    s.t. sum(w)=equity_budget, 0<=w_i<=MAX_POSITION, EM combined <= MAX_EM.
    """
    n = len(tickers)
    if n == 0 or equity_budget <= 0:
        return pd.Series(0.0, index=tickers)

    mu_v = mu.values
    # If no positive expected return, fall back to min-variance
    use_min_var = (mu_v <= 0).all()

    def neg_sharpe(w):
        port_ret = w @ mu_v
        port_var = w @ cov @ w
        if port_var <= 0:
            return 0.0
        return -port_ret / np.sqrt(port_var)

    def port_var_obj(w):
        return w @ cov @ w

    obj = port_var_obj if use_min_var else neg_sharpe

    # Constraints
    cons = [{"type": "eq", "fun": lambda w: w.sum() - equity_budget}]
    em_idx = [i for i, t in enumerate(tickers) if t in EM_EQUITY]
    if em_idx:
        cons.append({"type": "ineq",
                     "fun": lambda w: MAX_EM - w[em_idx].sum()})

    bounds = [(0.0, MAX_POSITION)] * n
    w0 = np.full(n, equity_budget / n)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    w = res.x if res.success else w0
    w = np.clip(w, 0.0, MAX_POSITION)
    # Renormalise to budget
    if w.sum() > 0:
        w = w * (equity_budget / w.sum())
    return pd.Series(w, index=tickers)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 4 — Bond duration by rates regime
# ══════════════════════════════════════════════════════════════════════════════

def bond_weights(p_bear_global: float, p_bear_rates: float) -> pd.Series:
    """
    Duration governed primarily by the rates regime:
      rates-bear (rising yields)  -> short duration regardless of equity
      rates-benign + equity-bear  -> long duration (flight-to-quality)
      rates-benign + equity-calm  -> intermediate
    """
    if p_bear_rates >= 0.50:                      # rising-yield regime
        alloc = {"VGSH": 0.40, "SHV": 0.30, "BIL": 0.30}
    else:                                          # benign / falling yields
        if p_bear_global < 0.35:
            alloc = {"VGSH": 0.40, "VGIT": 0.40, "SHV": 0.20}
        elif p_bear_global < 0.65:
            alloc = {"VGIT": 0.40, "VGLT": 0.30, "VGSH": 0.30}
        else:
            alloc = {"EDV": 0.40, "VGLT": 0.40, "VGIT": 0.20}
    return pd.Series(alloc)


# ══════════════════════════════════════════════════════════════════════════════
# Recovery trigger
# ══════════════════════════════════════════════════════════════════════════════

def compute_recovery_flags(month_ends: pd.DatetimeIndex) -> pd.Series:
    """True when V-shaped recovery is confirmed (release de-risking)."""
    # SPY price + VIX
    spy = pd.read_csv(RAW / "SPY.csv",
                      usecols=["m_date","m_close_dividend_and_split_adjusted"],
                      parse_dates=["m_date"])
    spy = (spy.rename(columns={"m_date":"date",
                                "m_close_dividend_and_split_adjusted":"px"})
              .dropna().set_index("date").sort_index()["px"])
    vix = pd.read_csv(RAW / "VIX_History.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"]

    ma200 = spy.rolling(200, min_periods=100).mean()
    ret20 = spy.pct_change(20)

    flags = {}
    for dt in month_ends:
        px_hist = spy.loc[:dt]
        if px_hist.empty:
            flags[dt] = False
            continue
        last = px_hist.index[-1]
        cond_ret = ret20.loc[last] > REC_RET_20D if last in ret20.index else False
        cond_ma  = (spy.loc[last] > ma200.loc[last]) if last in ma200.index and not np.isnan(ma200.loc[last]) else False
        v_hist   = vix.loc[:dt]
        cond_vix = (v_hist.iloc[-1] < REC_VIX_MAX) if len(v_hist) else False
        flags[dt] = bool(cond_ret and cond_ma and cond_vix)
    return pd.Series(flags)


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio build
# ══════════════════════════════════════════════════════════════════════════════

def build_portfolio(prices, returns, alpha, p_bull_assets,
                    p_bear_global, p_bear_rates, recovery_flags):
    month_ends = (alpha.index
                  .intersection(p_bear_global.index)
                  .intersection(p_bear_rates.index))
    rows = []
    for dt in month_ends:
        pbg_raw = float(p_bear_global.loc[dt])
        pbr     = float(p_bear_rates.loc[dt])
        # Recovery override
        pbg = min(pbg_raw, REC_PBEAR_CAP) if recovery_flags.get(dt, False) else pbg_raw

        equity_budget = EQUITY_MAX - pbg * (EQUITY_MAX - EQUITY_MIN)
        bond_budget   = 1.0 - equity_budget

        # ── Layer 2/3: regime-gated alpha + optimizer ────────────────────────
        a = alpha.loc[dt].dropna()
        # per-asset bull prob (fallback to global bull if asset history missing)
        if dt in p_bull_assets.index:
            pbull = p_bull_assets.loc[dt].reindex(a.index)
        else:
            pbull = pd.Series(np.nan, index=a.index)
        pbull = pbull.fillna(1.0 - pbg)          # fallback: global bull prob
        mu = (a * pbull).dropna()                # regime-gated expected return

        tickers = [t for t in mu.index if t in returns.columns]
        mu = mu[tickers]

        # Ledoit-Wolf covariance on trailing window
        hist = returns[tickers].loc[:dt].iloc[-COV_LOOKBACK:].dropna(axis=1, how="any")
        tickers = list(hist.columns)
        mu = mu.reindex(tickers).fillna(0.0)
        if len(tickers) >= 5 and len(hist) >= 60:
            lw = LedoitWolf().fit(hist.values)
            cov = lw.covariance_ * ANN
        else:
            cov = np.diag(np.ones(len(tickers)) * 0.04)

        eq_w = max_sharpe_weights(mu, cov, tickers, equity_budget)

        # ── Layer 4: bonds ───────────────────────────────────────────────────
        bw = bond_weights(pbg, pbr) * bond_budget

        # ── Combine ───────────────────────────────────────────────────────────
        w = pd.Series(0.0, index=ALL_ASSETS)
        for t, v in eq_w.items():
            if t in w.index: w[t] += v
        for t, v in bw.items():
            if t in w.index: w[t] += v

        # ── Vol targeting (whole portfolio, Ledoit-Wolf) ─────────────────────
        active = [t for t in w.index if w[t] > 0 and t in returns.columns]
        h = returns[active].loc[:dt].iloc[-VOL_LOOKBACK:].dropna(axis=1, how="any")
        active = list(h.columns)
        if len(active) >= 2 and len(h) >= 30:
            lw = LedoitWolf().fit(h.values)
            wv = w[active].values
            port_vol = np.sqrt(wv @ (lw.covariance_ * ANN) @ wv)
            scale = min(1.0, VOL_TARGET / max(port_vol, 1e-4))
        else:
            scale = 1.0
        w = w * scale
        resid = 1.0 - w.sum()
        w["BIL"] = w.get("BIL", 0.0) + max(resid, 0.0)

        w.name = dt
        rows.append(w)

    W = pd.DataFrame(rows)
    W.index.name = "date"
    return W


def compute_returns(weights, returns):
    dates = sorted(weights.index)
    chunks = []
    for i, dt in enumerate(dates):
        w = weights.loc[dt]
        nxt = dates[i+1] if i+1 < len(dates) else returns.index[-1]
        mask = (returns.index > dt) & (returns.index <= nxt)
        daily = returns.loc[mask]
        if daily.empty:
            continue
        common = [t for t in w.index if t in returns.columns and w[t] > 0]
        ws = w[common]; ws /= ws.sum()
        chunks.append(daily[common].fillna(0.0).dot(ws))
    return pd.concat(chunks).sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("V2 Strategy — Layers 2-4")
    print("=" * 64)

    prices  = pd.read_parquet(PROC / "prices.parquet")
    returns = pd.read_parquet(PROC / "returns.parquet")

    p_global = pd.read_parquet(V2 / "p_bear_global_monthly.parquet").iloc[:, 0]
    p_rates  = pd.read_parquet(V2 / "p_bear_rates_monthly.parquet").iloc[:, 0]
    p_assets = pd.read_parquet(V2 / "p_bull_assets_monthly.parquet")

    month_ends = prices.loc[BACKTEST_START:].resample("ME").last().index
    month_ends = pd.DatetimeIndex([d for d in month_ends if d in prices.index])

    print(f"[OK] rebalance dates: {len(month_ends)}")
    print("[1/3] Computing multi-factor alpha...")
    alpha = compute_alpha_scores(prices, returns, month_ends)

    print("[2/3] Computing recovery-trigger flags...")
    rec = compute_recovery_flags(month_ends)
    print(f"      recovery months: {int(rec.sum())}")

    print("[3/3] Building portfolio (optimizer)...")
    # align regime indices to month_ends
    p_global = p_global.reindex(month_ends, method="ffill")
    p_rates  = p_rates.reindex(month_ends, method="ffill")

    W = build_portfolio(prices, returns, alpha, p_assets,
                        p_global, p_rates, rec)
    R = compute_returns(W, returns)

    W.to_parquet(V2 / "v2_weights.parquet")
    R.to_frame("strategy").to_parquet(V2 / "v2_returns.parquet")

    # Quick preview
    bm = returns["AOA"].reindex(R.index).fillna(0.0)
    def stats(r):
        c = r.mean()*ANN; v = r.std()*np.sqrt(ANN)
        dd = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
        return c, v, c/v, dd
    sc, sv, ss, sdd = stats(R)
    bc, bv, bs, bdd = stats(bm)
    print(f"\n  Strategy: CAGR={sc:.1%} Vol={sv:.1%} Sharpe={ss:.2f} MaxDD={sdd:.1%}")
    print(f"  AOA     : CAGR={bc:.1%} Vol={bv:.1%} Sharpe={bs:.2f} MaxDD={bdd:.1%}")
    print(f"\n[SAVED] v2_weights.parquet, v2_returns.parquet → {V2}")


if __name__ == "__main__":
    main()
