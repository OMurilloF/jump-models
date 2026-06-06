"""
V2 Backtest Analytics & V1/V2/AOA Comparison
=============================================
Produces a comparison report and charts for V2 vs V1 vs AOA, including:
  - Full performance stats (CAGR, Sharpe, Sortino, Calmar, MaxDD)
  - IS/OOS split
  - Year-by-year
  - Deflated Sharpe Ratio (Lopez de Prado) given the number of configs tried

Outputs:
  Data/outputs/v2_comparison_report.txt
  Data/outputs/plots/v2_*.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

PROC = ROOT / "Data" / "processed"
V2   = PROC / "v2"
TUNE = ROOT / "Data" / "tuning"
OUT  = ROOT / "Data" / "outputs"
PLOT = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True); PLOT.mkdir(parents=True, exist_ok=True)
ANN = 252

plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"text.usetex":False,
                     "axes.titlesize":14,"axes.labelsize":12,"legend.fontsize":10})


def stats(r, bm=None):
    cagr = r.mean()*ANN; vol = r.std()*np.sqrt(ANN)
    sharpe = cagr/vol if vol>0 else np.nan
    downside = r.clip(upper=0); dvol = downside.std()*np.sqrt(ANN)
    sortino = cagr/dvol if dvol>0 else np.nan
    cum = (1+r).cumprod(); dd = (cum/cum.cummax()-1).min()
    calmar = cagr/abs(dd) if dd!=0 else np.nan
    m = r.resample("ME").sum()
    out = dict(cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
               calmar=calmar, maxdd=dd, cumret=cum.iloc[-1]-1,
               winrate=(m>0).mean())
    if bm is not None:
        b = bm.reindex(r.index).fillna(0.0); ex = r-b
        te = ex.std()*np.sqrt(ANN)
        out.update(excess=ex.mean()*ANN, te=te,
                   ir=(ex.mean()*ANN)/te if te>0 else np.nan,
                   beta=r.cov(b)/b.var(), corr=r.corr(b))
    return out


def deflated_sharpe(r, n_trials, sr_benchmark=0.0):
    """
    Lopez de Prado Deflated Sharpe Ratio.
    Adjusts observed SR for the number of configurations tried and for
    non-normal returns (skew/kurtosis).
    """
    sr   = (r.mean()/r.std()) * np.sqrt(ANN)        # annualised SR
    sr_d = r.mean()/r.std()                          # per-period SR
    T    = len(r)
    g3   = r.skew()
    g4   = r.kurtosis() + 3.0                        # raw kurtosis

    # Expected max SR across n_trials (variance of trial SRs ~ 1/T approx)
    e_max = np.sqrt(1.0/T) * ((1-np.euler_gamma)*norm.ppf(1-1.0/n_trials)
                              + np.euler_gamma*norm.ppf(1-1.0/(n_trials*np.e)))
    # PSR / DSR
    num = (sr_d - e_max) * np.sqrt(T-1)
    den = np.sqrt(1 - g3*sr_d + ((g4-1)/4.0)*sr_d**2)
    dsr = norm.cdf(num/den) if den>0 else np.nan
    return sr, e_max*np.sqrt(ANN), dsr


def main():
    returns = pd.read_parquet(PROC / "returns.parquet")
    v2 = pd.read_parquet(V2 / "v2_returns.parquet")["strategy"]
    v1 = pd.read_parquet(TUNE / "final_portfolio_returns.parquet")["strategy"]
    bm = returns["AOA"]

    # Align all to common window
    idx = v2.index.intersection(v1.index).intersection(bm.index)
    v2, v1, bm = v2.loc[idx], v1.loc[idx], bm.loc[idx]

    L = []
    def add(s=""): L.append(s)
    sep = "="*70
    add(sep); add("V2 vs V1 vs AOA — COMPARISON REPORT")
    add(f"Period: {idx.min().date()} to {idx.max().date()}"); add(sep)

    # Overall
    add("\n1. OVERALL PERFORMANCE")
    add("-"*70)
    hdr = f"  {'Metric':<16}{'V2':>12}{'V1':>12}{'AOA':>12}"
    add(hdr); add("  "+"-"*52)
    s2, s1, sb = stats(v2, bm), stats(v1, bm), stats(bm)
    def row(name, key, fmt):
        add(f"  {name:<16}{fmt(s2[key]):>12}{fmt(s1[key]):>12}{fmt(sb.get(key,np.nan)):>12}")
    pct = lambda x: f"{x*100:.1f}%"
    num = lambda x: f"{x:.2f}"
    row("CAGR","cagr",pct); row("Volatility","vol",pct)
    row("Sharpe","sharpe",num); row("Sortino","sortino",num)
    row("Calmar","calmar",num); row("Max Drawdown","maxdd",pct)
    row("Cumulative","cumret",pct); row("Monthly Win%","winrate",pct)
    add(f"  {'Excess vs AOA':<16}{pct(s2['excess']):>12}{pct(s1['excess']):>12}{'-':>12}")
    add(f"  {'Info Ratio':<16}{num(s2['ir']):>12}{num(s1['ir']):>12}{'-':>12}")
    add(f"  {'Tracking Err':<16}{pct(s2['te']):>12}{pct(s1['te']):>12}{'-':>12}")
    add(f"  {'Beta vs AOA':<16}{num(s2['beta']):>12}{num(s1['beta']):>12}{'-':>12}")

    # IS/OOS
    add("\n\n2. IS / OOS SPLIT")
    add("-"*70)
    for lbl, a, b in [("IS 2010-2019","2010","2019"),("OOS 2020-2025","2020","2025")]:
        x2, x1, xb = stats(v2.loc[a:b],bm.loc[a:b]), stats(v1.loc[a:b],bm.loc[a:b]), stats(bm.loc[a:b])
        add(f"\n  [{lbl}]")
        add(f"    V2 : CAGR={pct(x2['cagr'])}  Sharpe={x2['sharpe']:.2f}  "
            f"Sortino={x2['sortino']:.2f}  MaxDD={pct(x2['maxdd'])}  Excess={pct(x2['excess'])}")
        add(f"    V1 : CAGR={pct(x1['cagr'])}  Sharpe={x1['sharpe']:.2f}  "
            f"Sortino={x1['sortino']:.2f}  MaxDD={pct(x1['maxdd'])}  Excess={pct(x1['excess'])}")
        add(f"    AOA: CAGR={pct(xb['cagr'])}  Sharpe={xb['sharpe']:.2f}  "
            f"Sortino={xb['sortino']:.2f}  MaxDD={pct(xb['maxdd'])}")

    # Year by year
    add("\n\n3. ANNUAL RETURNS")
    add("-"*70)
    add(f"  {'Year':<6}{'V2':>10}{'V1':>10}{'AOA':>10}{'V2-AOA':>10}")
    add("  "+"-"*46)
    for yr in range(2010,2026):
        a2,a1,ab = v2.loc[str(yr)],v1.loc[str(yr)],bm.loc[str(yr)]
        if len(a2)==0: continue
        add(f"  {yr:<6}{a2.sum()*100:>9.1f}%{a1.sum()*100:>9.1f}%"
            f"{ab.sum()*100:>9.1f}%{(a2.sum()-ab.sum())*100:>+9.1f}%")

    # Deflated Sharpe
    add("\n\n4. DEFLATED SHARPE RATIO (Lopez de Prado)")
    add("-"*70)
    N_TRIALS = 160  # ~120 (grid1) + 36 (grid2) + a few v2 variants
    sr, e_max, dsr = deflated_sharpe(v2, N_TRIALS)
    add(f"  Configs tried (approx)      : {N_TRIALS}")
    add(f"  Observed annualised SR (V2) : {sr:.2f}")
    add(f"  Expected max SR by chance   : {e_max:.2f}")
    add(f"  Deflated Sharpe (prob true) : {dsr:.1%}")
    add(f"  Interpretation: DSR > 95% => SR unlikely a multiple-testing artifact.")

    add(f"\n{sep}"); add("END"); add(sep)
    report = "\n".join(L)
    (OUT / "v2_comparison_report.txt").write_text(report)
    print(report)

    # ── Charts ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14,6))
    (100*(1+v2).cumprod()).plot(ax=ax,lw=2,label="V2",color="#2ca02c")
    (100*(1+v1).cumprod()).plot(ax=ax,lw=2,label="V1",color="#1f77b4")
    (100*(1+bm).cumprod()).plot(ax=ax,lw=2,label="AOA",color="#ff7f0e",ls="--")
    ax.set_title("Cumulative Performance: V2 vs V1 vs AOA (base=100)")
    ax.set_ylabel("Value"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOT/"v2_01_cumulative.png"); plt.close()

    def dds(r): c=(1+r).cumprod(); return c/c.cummax()-1
    fig, ax = plt.subplots(figsize=(14,6))
    (dds(v2)*100).plot(ax=ax,label="V2",color="#2ca02c",lw=1.2)
    (dds(bm)*100).plot(ax=ax,label="AOA",color="#ff7f0e",lw=1.2,ls="--")
    ax.fill_between(dds(v2).index, dds(v2)*100,0,color="#2ca02c",alpha=0.25)
    ax.set_title("Drawdown: V2 vs AOA"); ax.set_ylabel("Drawdown (%)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOT/"v2_02_drawdown.png"); plt.close()

    s_ann=v2.resample("YE").sum()*100; b_ann=bm.resample("YE").sum()*100
    v1_ann=v1.resample("YE").sum()*100; yrs=s_ann.index.year
    x=np.arange(len(yrs)); w=0.27
    fig,ax=plt.subplots(figsize=(14,6))
    ax.bar(x-w,s_ann.values,w,label="V2",color="#2ca02c")
    ax.bar(x,v1_ann.values,w,label="V1",color="#1f77b4")
    ax.bar(x+w,b_ann.values,w,label="AOA",color="#ff7f0e")
    ax.axhline(0,color="k",lw=0.8); ax.set_xticks(x); ax.set_xticklabels(yrs,rotation=45)
    ax.set_ylabel("Annual Return (%)"); ax.set_title("Annual Returns")
    ax.legend(); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOT/"v2_03_annual.png"); plt.close()

    print(f"\n[SAVED] report + 3 charts → {OUT}")


if __name__ == "__main__":
    main()
