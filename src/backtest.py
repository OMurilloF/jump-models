"""
Phase 5 — Backtest Analytics
==============================
Comprehensive performance attribution and risk analytics for the final
tuned strategy vs AOA benchmark.

Produces:
  Data/outputs/performance_report.txt   — full text report
  Data/outputs/plots/                   — 6 PNG charts
    01_cumulative_returns.png
    02_underwater.png
    03_rolling_sharpe.png
    04_regime_signal.png
    05_annual_returns.png
    06_regime_conditional.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "Data" / "outputs"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ANN = 252
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "font.size": 11,
    "text.usetex": False,
})


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    tune = ROOT / "Data" / "tuning"
    proc = ROOT / "Data" / "processed"

    strat_ret = pd.read_parquet(tune / "final_portfolio_returns.parquet")["strategy"]
    bm_ret    = pd.read_parquet(proc / "benchmark_returns.parquet")["AOA"]
    weights   = pd.read_parquet(tune / "final_portfolio_weights.parquet")
    p_bear_m  = pd.read_parquet(tune / "final_p_bear_monthly.parquet")["p_bear_eom"]
    p_bear_d  = pd.read_parquet(proc / "p_bear.parquet")["p_bear"]    # daily

    # Align all to common date range
    idx = strat_ret.index.intersection(bm_ret.index)
    strat_ret = strat_ret.loc[idx]
    bm_ret    = bm_ret.loc[idx]

    return strat_ret, bm_ret, weights, p_bear_m, p_bear_d


# ══════════════════════════════════════════════════════════════════════════════
# Performance statistics
# ══════════════════════════════════════════════════════════════════════════════

def perf_stats(r: pd.Series, bm: pd.Series = None, label: str = "") -> dict:
    cagr    = r.mean() * ANN
    vol     = r.std() * np.sqrt(ANN)
    sharpe  = cagr / vol if vol > 0 else np.nan
    cum     = (1 + r).cumprod()
    dd_ser  = cum / cum.cummax() - 1
    maxdd   = dd_ser.min()
    calmar  = cagr / abs(maxdd) if maxdd != 0 else np.nan
    cum_ret = cum.iloc[-1] - 1

    # Monthly win rate
    monthly_r = r.resample("ME").sum()
    win_rate  = (monthly_r > 0).mean()

    stats = dict(
        label=label, cagr=cagr, vol=vol, sharpe=sharpe,
        maxdd=maxdd, calmar=calmar, cum_ret=cum_ret, win_rate=win_rate,
    )

    if bm is not None:
        bm_a = bm.reindex(r.index).fillna(0.0)
        excess = r - bm_a
        te     = excess.std() * np.sqrt(ANN)
        ir     = (excess.mean() * ANN) / te if te > 0 else np.nan
        beta   = r.cov(bm_a) / bm_a.var()
        corr   = r.corr(bm_a)
        stats.update(dict(excess_cagr=excess.mean()*ANN, te=te, ir=ir, beta=beta, corr=corr))

    return stats


def fmt_pct(x, decimals=1): return f"{x*100:+.{decimals}f}%"
def fmt_pct0(x): return f"{x*100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# Drawdown table
# ══════════════════════════════════════════════════════════════════════════════

def drawdown_table(r: pd.Series, top_n: int = 8) -> pd.DataFrame:
    cum     = (1 + r).cumprod()
    peak    = cum.cummax()
    dd_ser  = (cum / peak - 1)
    in_dd   = dd_ser < -0.03   # threshold: 3%

    rows = []
    start = None
    for i, (dt, v) in enumerate(in_dd.items()):
        if v and start is None:
            start = dt
        elif not v and start is not None:
            window = dd_ser.loc[start:]
            trough_dt = window[:dt].idxmin()
            depth     = dd_ser.loc[trough_dt]
            duration  = (trough_dt - start).days
            recovery  = (dt - trough_dt).days
            rows.append(dict(
                start=start.date(), trough=trough_dt.date(), end=dt.date(),
                depth=depth, duration_days=duration, recovery_days=recovery
            ))
            start = None
    if start is not None:  # still in drawdown at end
        window    = dd_ser.loc[start:]
        trough_dt = window.idxmin()
        depth     = dd_ser.loc[trough_dt]
        rows.append(dict(
            start=start.date(), trough=trough_dt.date(), end="ongoing",
            depth=depth, duration_days=(trough_dt - start).days, recovery_days=None
        ))

    df = pd.DataFrame(rows).sort_values("depth").head(top_n)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Attribution: equity vs bonds vs cash
# ══════════════════════════════════════════════════════════════════════════════

FIXED_INCOME = ["EDV", "VGLT", "VGIT", "VGSH", "SHV"]
CASH         = ["BIL"]
EM_CREDIT    = ["EMHY"]
TIER1_EQUITY = [
    "VGT","VHT","VFH","VCR","VDC","VIS","VAW","VDE","VPU","VOX","VNQ",
    "EWJ","EWG","EWU","EWL","EWQ","EWI","EWP","EWD","EWN","EWO","EWK",
    "EWS","EWA","EWC","EWT","EWH",
    "EWZ","EWW","EWY","EWM","EZA","THD","TUR","EPHE","GXC","ECH","EPU",
]


def sleeve_attribution(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Monthly contribution from equity, bonds, and cash sleeves."""
    fi_names   = FIXED_INCOME + EM_CREDIT
    cash_names = CASH
    eq_names   = TIER1_EQUITY

    dates  = sorted(weights.index)
    rows   = []
    for i, dt in enumerate(dates):
        w       = weights.loc[dt]
        next_dt = dates[i+1] if i+1 < len(dates) else returns.index[-1]
        mask    = (returns.index > dt) & (returns.index <= next_dt)
        daily   = returns.loc[mask]
        if daily.empty:
            continue

        def sleeve_ret(names):
            tickers = [t for t in names if t in w.index and t in returns.columns and w[t] > 0]
            if not tickers:
                return 0.0
            w_s = w[tickers]
            return daily[tickers].fillna(0.0).dot(w_s).sum()

        rows.append(dict(
            date=next_dt,
            equity=sleeve_ret(eq_names),
            bonds=sleeve_ret(fi_names),
            cash=sleeve_ret(cash_names),
        ))

    return pd.DataFrame(rows).set_index("date")


# ══════════════════════════════════════════════════════════════════════════════
# Regime-conditional performance
# ══════════════════════════════════════════════════════════════════════════════

def regime_conditional(r: pd.Series, p_bear_m: pd.Series) -> pd.DataFrame:
    """
    Monthly returns split by p_bear regime buckets.
    Uses end-of-month p_bear to label each month (naturally binary from CJM).
    """
    monthly_r = r.resample("ME").sum()
    # Align p_bear_m to monthly_r dates by forward-filling
    pb = p_bear_m.reindex(monthly_r.index, method="ffill")

    rows = []
    buckets = [
        ("Bull (p_bear < 0.25)",      pb < 0.25),
        ("Transition (0.25 – 0.50)",  (pb >= 0.25) & (pb < 0.50)),
        ("Bear (p_bear >= 0.50)",      pb >= 0.50),
    ]
    for label, mask in buckets:
        sub = monthly_r.loc[mask]
        if len(sub) < 3:
            continue
        ann_ret = sub.mean() * 12
        ann_vol = sub.std() * np.sqrt(12)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rows.append(dict(
            regime=label, n_months=len(sub),
            ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
            pct_time=len(sub) / len(monthly_r),
        ))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

STRAT_COLOR = "#1f77b4"
BM_COLOR    = "#ff7f0e"
BEAR_COLOR  = "#d62728"


def plot_cumulative(strat: pd.Series, bm: pd.Series):
    fig, ax = plt.subplots(figsize=(14, 6))
    (100 * (1 + strat).cumprod()).plot(ax=ax, color=STRAT_COLOR, lw=2, label="Strategy")
    (100 * (1 + bm).cumprod()).plot(ax=ax, color=BM_COLOR, lw=2, label="AOA", linestyle="--")
    ax.axhline(100, color="gray", lw=0.8, linestyle=":")
    ax.set_title("Cumulative Performance (base = 100)")
    ax.set_ylabel("Portfolio Value")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_cumulative_returns.png")
    plt.close()


def plot_underwater(strat: pd.Series, bm: pd.Series):
    def dd_series(r):
        cum = (1+r).cumprod(); return cum/cum.cummax()-1

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, r, label, color in zip(
            axes, [strat, bm], ["Strategy", "AOA"], [STRAT_COLOR, BM_COLOR]):
        d = dd_series(r)
        ax.fill_between(d.index, d*100, 0, color=color, alpha=0.4)
        ax.plot(d.index, d*100, color=color, lw=1)
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"{label} — Drawdown")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_underwater.png")
    plt.close()


def plot_rolling_sharpe(strat: pd.Series, bm: pd.Series, window: int = 252):
    def rolling_sharpe(r, w):
        roll_mean = r.rolling(w).mean() * ANN
        roll_std  = r.rolling(w).std() * np.sqrt(ANN)
        return roll_mean / roll_std

    fig, ax = plt.subplots(figsize=(14, 5))
    rolling_sharpe(strat, window).plot(ax=ax, color=STRAT_COLOR, lw=1.5, label="Strategy")
    rolling_sharpe(bm, window).plot(ax=ax, color=BM_COLOR, lw=1.5, label="AOA", linestyle="--")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="green", lw=0.6, linestyle=":")
    ax.set_title(f"Rolling {window//21}-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_rolling_sharpe.png")
    plt.close()


def plot_regime_signal(p_bear_m: pd.Series, weights: pd.DataFrame):
    eq_names = TIER1_EQUITY
    eq_w = weights[[c for c in eq_names if c in weights.columns]].sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1 = axes[0]
    ax1.fill_between(p_bear_m.index, p_bear_m, 0, color=BEAR_COLOR, alpha=0.3, step="post")
    ax1.plot(p_bear_m.index, p_bear_m, color=BEAR_COLOR, lw=1, drawstyle="steps-post")
    ax1.axhline(0.5, color="gray", lw=0.8, linestyle=":")
    ax1.set_ylabel("Bear Probability"); ax1.set_title("CJM Bear Probability (p_bear)")
    ax1.set_ylim(0, 1); ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(eq_w.index, eq_w*100, 0, color=STRAT_COLOR, alpha=0.3, step="post")
    ax2.plot(eq_w.index, eq_w*100, color=STRAT_COLOR, lw=1, drawstyle="steps-post")
    ax2.set_ylabel("Equity Allocation (%)"); ax2.set_title("Portfolio Equity Weight")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_regime_signal.png")
    plt.close()


def plot_annual_returns(strat: pd.Series, bm: pd.Series):
    s_ann = strat.resample("YE").sum() * 100
    b_ann = bm.resample("YE").sum() * 100
    years = s_ann.index.year

    x = np.arange(len(years)); w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars_s = ax.bar(x - w/2, s_ann.values, w, label="Strategy", color=STRAT_COLOR, alpha=0.85)
    bars_b = ax.bar(x + w/2, b_ann.values, w, label="AOA",      color=BM_COLOR,    alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Annual Return (%)"); ax.set_title("Annual Returns: Strategy vs AOA")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_annual_returns.png")
    plt.close()


def plot_regime_conditional(rc_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728"]
    x      = np.arange(len(rc_df))

    ax = axes[0]
    bars = ax.bar(x, rc_df["ann_ret"]*100, color=colors[:len(rc_df)], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(rc_df["regime"], rotation=15, ha="right")
    ax.set_ylabel("Ann. Return (%)"); ax.set_title("Strategy Return by Regime")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    bars2 = ax2.bar(x, rc_df["sharpe"], color=colors[:len(rc_df)], alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(rc_df["regime"], rotation=15, ha="right")
    ax2.set_ylabel("Sharpe Ratio"); ax2.set_title("Sharpe Ratio by Regime")
    ax2.grid(axis="y", alpha=0.3)

    # annotate % time in each regime
    for ax_ in [ax, ax2]:
        for i, (_, row) in enumerate(rc_df.iterrows()):
            ax_.text(i, ax_.get_ylim()[0] * 0.05 if ax_.get_ylim()[0] < 0 else 0.3,
                     f"{row.pct_time:.0%}", ha="center",
                     va="bottom", fontsize=9, color="black")

    plt.suptitle("Regime-Conditional Performance (% labels = time in regime)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_regime_conditional.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Text report
# ══════════════════════════════════════════════════════════════════════════════

def build_report(strat, bm, weights, p_bear_m, p_bear_d, returns) -> str:
    lines = []
    sep   = "=" * 70

    def add(s=""): lines.append(s)

    add(sep)
    add("STRATEGY BACKTEST REPORT")
    add(f"Period: {strat.index.min().date()} to {strat.index.max().date()}")
    add(f"Config: lambda=25  equity_min=50%  equity_max=80%  yc_bonds=True  ma_filter=False")
    add(sep)

    # ── 1. Overall performance ─────────────────────────────────────────────
    add("\n1. OVERALL PERFORMANCE")
    add("-" * 40)
    for label, r, use_bm in [("Strategy", strat, True), ("AOA (Benchmark)", bm, False)]:
        s = perf_stats(r, bm if use_bm else None, label)
        add(f"\n  {label}")
        add(f"    CAGR           : {fmt_pct0(s['cagr'])}")
        add(f"    Volatility     : {fmt_pct0(s['vol'])}")
        add(f"    Sharpe Ratio   : {s['sharpe']:.2f}")
        add(f"    Max Drawdown   : {fmt_pct(s['maxdd'])}")
        add(f"    Calmar Ratio   : {s['calmar']:.2f}")
        add(f"    Cumulative Ret : {fmt_pct0(s['cum_ret'])}")
        add(f"    Monthly Win %  : {s['win_rate']:.0%}")
        if use_bm:
            add(f"    Excess Return  : {fmt_pct(s['excess_cagr'])}")
            add(f"    Tracking Error : {fmt_pct0(s['te'])}")
            add(f"    Info Ratio     : {s['ir']:.2f}")
            add(f"    Beta vs AOA    : {s['beta']:.2f}")
            add(f"    Correlation    : {s['corr']:.2f}")

    # IS / OOS split
    add("\n  --- IS / OOS Split ---")
    for period, s, e in [("IS (2010-2019)", "2010", "2019"),
                          ("OOS(2020-2025)", "2020", "2025")]:
        sr = strat.loc[s:e]; br = bm.loc[s:e]
        ss = perf_stats(sr, br); bs = perf_stats(br)
        add(f"\n  {period}")
        add(f"    Strategy: CAGR={fmt_pct0(ss['cagr'])}  Vol={fmt_pct0(ss['vol'])}  "
            f"Sharpe={ss['sharpe']:.2f}  MaxDD={fmt_pct(ss['maxdd'])}  "
            f"Excess={fmt_pct(ss['excess_cagr'])}")
        add(f"    AOA:      CAGR={fmt_pct0(bs['cagr'])}  Vol={fmt_pct0(bs['vol'])}  "
            f"Sharpe={bs['sharpe']:.2f}  MaxDD={fmt_pct(bs['maxdd'])}")

    # ── 2. Drawdown table ──────────────────────────────────────────────────
    add(f"\n\n2. TOP DRAWDOWNS (Strategy)")
    add("-" * 40)
    ddt = drawdown_table(strat, top_n=8)
    add(f"  {'Start':<12} {'Trough':<12} {'End':<12} {'Depth':>8} {'Days to Trough':>15} {'Recovery':>10}")
    add("  " + "-" * 70)
    for _, row in ddt.iterrows():
        rec = f"{row.recovery_days:.0f}d" if row.recovery_days is not None else "ongoing"
        add(f"  {str(row.start):<12} {str(row.trough):<12} {str(row.end):<12} "
            f"{row.depth*100:>7.1f}%  {row.duration_days:>14}d  {rec:>10}")

    # ── 3. Annual returns ─────────────────────────────────────────────────
    add(f"\n\n3. ANNUAL RETURNS")
    add("-" * 40)
    add(f"  {'Year':<6} {'Strategy':>10} {'AOA':>10} {'Excess':>10} {'p_bear':>8}")
    add("  " + "-" * 46)
    for yr in range(2010, 2026):
        sr = strat.loc[str(yr)]; br = bm.loc[str(yr)]
        if len(sr) == 0: continue
        pb = p_bear_m[p_bear_m.index.year == yr].mean()
        add(f"  {yr:<6} {sr.sum()*100:>9.1f}%  {br.sum()*100:>8.1f}%  "
            f"{(sr.sum()-br.sum())*100:>+9.1f}%  {pb:>7.3f}")

    # ── 4. Regime-conditional performance ─────────────────────────────────
    add(f"\n\n4. REGIME-CONDITIONAL PERFORMANCE (Strategy)")
    add("-" * 40)
    rc = regime_conditional(strat, p_bear_m)
    add(f"  {'Regime':<30} {'Months':>7} {'Time':>6} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>8}")
    add("  " + "-" * 67)
    for _, row in rc.iterrows():
        add(f"  {row.regime:<30} {row.n_months:>7.0f} {row.pct_time:>5.0%}  "
            f"{row.ann_ret*100:>7.1f}%  {row.ann_vol*100:>6.1f}%  {row.sharpe:>7.2f}")

    # ── 5. Sleeve attribution ─────────────────────────────────────────────
    add(f"\n\n5. SLEEVE ATTRIBUTION (annualised contribution)")
    add("-" * 40)
    slv = sleeve_attribution(weights, returns)
    slv_ann = slv.mean() * 12 * 100  # monthly data → annualise by ×12
    total   = strat.mean() * ANN * 100
    add(f"  Total strategy return (ann)  : {total:+.2f}%")
    add(f"  Equity sleeve contribution   : {slv_ann['equity']:+.2f}%")
    add(f"  Bond sleeve contribution     : {slv_ann['bonds']:+.2f}%")
    add(f"  Cash (BIL) contribution      : {slv_ann['cash']:+.2f}%")
    add(f"  [Note: sum may differ from total due to scaling/residuals]")

    # Monthly equity weight stats
    eq_names = [c for c in TIER1_EQUITY if c in weights.columns]
    avg_eq   = weights[eq_names].sum(axis=1).mean()
    avg_fi   = weights[[c for c in FIXED_INCOME if c in weights.columns]].sum(axis=1).mean()
    avg_cash = weights["BIL"].mean() if "BIL" in weights.columns else 0
    add(f"\n  Avg monthly equity allocation: {avg_eq:.1%}")
    add(f"  Avg monthly bonds allocation : {avg_fi:.1%}")
    add(f"  Avg monthly cash allocation  : {avg_cash:.1%}")

    # ── 6. Key risk metrics ───────────────────────────────────────────────
    add(f"\n\n6. RISK METRICS")
    add("-" * 40)
    monthly = strat.resample("ME").sum()
    bm_m    = bm.resample("ME").sum().reindex(monthly.index).fillna(0)
    worst5  = monthly.nsmallest(5)
    add(f"  5 Worst Months (Strategy):")
    for dt, v in worst5.items():
        bv = bm_m.get(dt, 0.0)
        add(f"    {dt.strftime('%Y-%m')}: {v*100:+.1f}%  (AOA: {bv*100:+.1f}%)")

    best5 = monthly.nlargest(5)
    add(f"  5 Best Months (Strategy):")
    for dt, v in best5.items():
        bv = bm_m.get(dt, 0.0)
        add(f"    {dt.strftime('%Y-%m')}: {v*100:+.1f}%  (AOA: {bv*100:+.1f}%)")

    skew = monthly.skew()
    kurt = monthly.kurtosis()
    add(f"\n  Return skewness: {skew:+.2f}  (positive = right tail favourable)")
    add(f"  Excess kurtosis: {kurt:+.2f}  (fat tails if > 0)")

    add(f"\n{sep}")
    add("END OF REPORT")
    add(sep)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 5 — Backtest Analytics")
    print("=" * 70)

    strat, bm, weights, p_bear_m, p_bear_d = load_all()
    returns = pd.read_parquet(ROOT / "Data" / "processed" / "returns.parquet")

    print("[1/7] Generating cumulative return chart...")
    plot_cumulative(strat, bm)

    print("[2/7] Generating underwater (drawdown) chart...")
    plot_underwater(strat, bm)

    print("[3/7] Generating rolling Sharpe chart...")
    plot_rolling_sharpe(strat, bm)

    print("[4/7] Generating regime signal chart...")
    plot_regime_signal(p_bear_m, weights)

    print("[5/7] Generating annual returns chart...")
    plot_annual_returns(strat, bm)

    print("[6/7] Generating regime-conditional chart...")
    rc_df = regime_conditional(strat, p_bear_m)
    plot_regime_conditional(rc_df)

    print("[7/7] Building performance report...")
    report = build_report(strat, bm, weights, p_bear_m, p_bear_d, returns)
    report_path = OUT_DIR / "performance_report.txt"
    report_path.write_text(report)

    print("\n" + report)
    print(f"\n[SAVED] Report   → {report_path}")
    print(f"[SAVED] 6 plots  → {PLOT_DIR}")


if __name__ == "__main__":
    main()
