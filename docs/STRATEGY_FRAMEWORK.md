# Tactical Long-Only ETF Strategy Framework
## Jump-Model Regime Overlay on a Global Multi-Asset Universe

**Version:** 2.0 (Long-Only Revision)  
**Date:** 2026-06-06  
**Benchmark:** AOA (iShares Core Aggressive Allocation ETF)  
**Risk Budget:** ~35% Maximum Drawdown  
**Rebalance Frequency:** Monthly  

---

## 1. Change Log: Long/Short → Long-Only

The strategy was originally designed with four sleeves that included a cross-sectional
long/short momentum overlay and a time-series trend / CTA sleeve. With the mandate
revised to **long-only**, several components change while the core regime-detection
engine (Statistical Jump Models) is preserved.

| Component | L/S Original | Long-Only Revision | Rationale |
|-----------|-------------|-------------------|-----------|
| Cross-sectional momentum | Long top decile / Short bottom decile | Long top quintile, zero-weight bottom quintile | Cannot hold negative positions; losers are simply excluded |
| CTA / trend sleeve | Long-short futures overlay | Replaced by a duration-ladder rotation within fixed income | No futures access assumed; trend signal repurposed as bond duration tilting |
| Gross exposure | ~150–200% (leveraged L/S) | 100% fully invested (long only) | No leverage; weights always sum to 1 |
| Short book alpha | Negative-alpha positions monetized | Lost — bottom-quintile exclusion captures only partial benefit | Key expected-return reduction; partially offset by tighter top-quintile concentration |
| Regime "risk-off" | Short beta via short book | Cash / short-duration Treasuries / BIL | Flight to safety implemented via defensive asset rotation |
| Carry cost | Borrow fees on short leg | Eliminated | Simplifies ops and reduces drag in flat markets |
| Drawdown control | Short book provides natural hedge | Regime overlay + Treasury ladder + vol targeting | Must work harder on defensive rotation |

### Expected Impact of Going Long-Only

| Dimension | Direction | Magnitude | Notes |
|-----------|-----------|-----------|-------|
| Annualized return | Lower | −1.5 to −3.0% | Loss of short-book alpha and leverage |
| Sharpe ratio | Slightly lower | −0.1 to −0.2 | Short book improved risk-adjusted returns |
| Max drawdown | Higher | +5 to +10% | No natural hedge during bear regimes |
| Turnover | Lower | −20 to −30% | No short leg to manage |
| Factor exposure | Higher net beta | +0.15 to +0.25 β | Always net long; regime overlay only reduces, never inverts |
| Benchmark tracking | Tighter | TE ≈ 8–14% vs 12–20% L/S | Stays closer to AOA; excess return compressed |

The regime overlay and defensive rotation remain the primary alpha sources. The strategy
outperforms AOA by **timing asset-class exposure** — overweighting momentum leaders in
bull regimes and rotating to Treasuries/cash in bear regimes — rather than through
short-book profits.

---

## 2. Available Data

All files are located at `Data/raw/Macro and Prices/`.

### 2a. Equity ETFs (54 names)
**US Sectors (11):** VGT, VHT, VFH, VCR, VDC, VIS, VAW, VDE, VPU, VOX, VNQ  
**International Developed (18):** EWJ, EWG, EWU, EWL, EWQ, EWI, EWP, EWD, EWN, EWO,
EWK, EWS, EWA, EWC, EWT, EWH, EDEN, EFNL, ENOR, EIRL, ENZL, EPOL, PGAL  
**Emerging Markets (25):** EWZ, EWW, EWY, EWM, EZA, THD, TUR, EIDO (IDX), EPHE, GXC,
INDA (PIN), ECH, EPU, ARGT, EIS, KSA, UAE, QAT, KWT, NGE, EGPT, COLO, VNM, GREK  

### 2b. Fixed Income ETFs (6 names)
EDV, VGLT, VGIT, VGSH, SHV, BIL  
*(Duration ladder: ~24yr → ~16yr → ~5.4yr → ~1.9yr → ~0.4yr → ~0.14yr)*

### 2c. EM Fixed Income (1)
EMHY

### 2d. Benchmark (1)
AOA

### 2e. Macro / Market Data
- `SPY.csv` — S&P 500 proxy (equity regime feature)
- `VIX_History.csv` — CBOE VIX (volatility regime feature)
- `Treasuries_Historical Data.csv` — Treasury yields curve
- `^IDCOTSTR.csv` — ICE BofA OAS spread index (credit stress feature)

---

## 3. Universe Segmentation for Long-Only Portfolio

### Tier 1 — Active Trading Universe (42 names)
Liquid, institutionally-capacity-appropriate names. These receive positive weight.

| Segment | Tickers | Max Weight |
|---------|---------|-----------|
| US Sectors | VGT VHT VFH VCR VDC VIS VAW VDE VPU VOX VNQ | 40% combined |
| Developed International | EWJ EWG EWU EWL EWQ EWI EWP EWD EWN EWO EWK EWS EWA EWC EWT EWH | 35% combined |
| Emerging Markets | EWZ EWW EWY EWM EZA THD TUR EPHE GXC ECH EPU | 20% combined |
| Fixed Income | EDV VGLT VGIT VGSH SHV BIL | 0–50% (regime-dependent) |
| EM Credit | EMHY | 5% max |

### Tier 2 — Monitor Only (illiquid / small AUM)
ARGT, EDEN, EFNL, ENOR, EIRL, ENZL, EPOL, PGAL, EIS, KSA, UAE, QAT, KWT, NGE, EGPT,
COLO, VNM, GREK, IDX, PIN — tracked for signal construction but receive zero weight
unless AUM > $500M threshold is crossed.

---

## 4. Strategy Architecture (4 Sleeves, Revised)

### Sleeve A — Smart Beta Core (40% base weight)
**Unchanged from L/S design.**  
Equal-risk-weighted basket of Tier-1 equity ETFs, rebalanced monthly.
Provides diversified equity beta with lower idiosyncratic risk than market-cap weighting.

- Weights: inverse-volatility weighted (60-day realized vol)
- Equity-only, always long
- Serves as the "equity engine" of the portfolio

### Sleeve B — Cross-Sectional Momentum, Long-Only (20% base weight)

**L/S → Long-Only change:** Previously long top decile / short bottom decile. Now:
- Rank all Tier-1 equity ETFs by 12-1 momentum (12-month return excluding last month)
- **Long top quintile** (≈8–9 names) with equal weight within sleeve
- **Zero weight** to bottom quintile — these are simply not held
- Middle 3 quintiles held at market weight (Sleeve A handles them)

**Why this preserves alpha:** The long leg of momentum has historically contributed
~60% of the total L/S momentum premium. We capture this without the short-book
infrastructure cost.

**What is lost:** The short leg captured ~40% of momentum alpha and also provided
a natural equity hedge. In bear regimes, losing momentum names fall faster —
this hedge is now absent. The regime overlay (Sleeve D) must compensate.

### Sleeve C — Duration Ladder / Fixed Income Rotation (20% base weight)
**Replaces the CTA/trend sleeve from the L/S design.**

In the L/S framework, a CTA sleeve used long-short trend following across futures.
Long-only eliminates that. Instead, Sleeve C dynamically rotates along the
Treasury duration ladder based on the regime signal:

| Regime State | Duration Tilt | Allocation |
|-------------|--------------|-----------|
| Bull (confident) | Short duration — avoid rate risk | 80% SHV/BIL, 20% VGSH |
| Bull (uncertain) | Intermediate | 40% VGIT, 40% VGSH, 20% SHV |
| Transition | Barbell | 25% EDV, 25% VGLT, 25% VGIT, 25% VGSH |
| Bear (uncertain) | Long duration — flight to safety | 40% EDV, 40% VGLT, 20% VGIT |
| Bear (confident) | Max duration + EMHY defensive | 50% EDV, 30% VGLT, 20% EMHY |

This sleeve now serves the role the short book served in bear markets: it
appreciates (or at least holds value) when equities fall, providing a natural
offset to Sleeve A/B drawdowns.

### Sleeve D — Jump Model Regime Overlay (weights all sleeves)
**Core engine — unchanged in design, revised in application.**

The Statistical Jump Model (CJM variant) generates a bear-market probability
`p_bear ∈ [0, 1]` from daily features of SPY/equity returns:
- EWM return over 3 horizons (21d, 63d, 126d)
- EWM downside deviation over 3 horizons
- EWM Sortino ratio over 3 horizons

**Bear probability → Portfolio weights (long-only):**

```
equity_weight = (1 - p_bear) × equity_target
bond_weight   = p_bear × bond_target + (1 - p_bear) × bond_floor
cash_weight   = max(0, 1 - equity_weight - bond_weight)
```

Where:
- `equity_target` = 80% (max equity in bull regime)
- `bond_target` = 60% (max bonds in bear regime)
- `bond_floor` = 10% (always hold some bonds)

**In L/S, bear regime meant:** increase short book, reduce net long beta  
**In long-only, bear regime means:** rotate from equities into Treasury ladder

The CJM probability is used continuously (not thresholded) to allow smooth,
gradual rotation rather than binary switches — this reduces turnover and
avoids whipsawing.

---

## 5. Portfolio Construction

### 5a. Weight Determination (Monthly)

```
Step 1: Compute p_bear from CJM.predict_online() on last available day
Step 2: Compute sleeve-level allocations from p_bear (Section 4)
Step 3: Within Sleeve A: inverse-vol weight Tier-1 equities
Step 4: Within Sleeve B: equal-weight top-quintile momentum names
Step 5: Within Sleeve C: duration-ladder weights per regime table
Step 6: Blend sleeves: A×w_A + B×w_B + C×w_C
Step 7: Apply position limits (below)
Step 8: Normalize weights to sum = 1.00
```

### 5b. Position Limits

| Constraint | Value |
|-----------|-------|
| Single name max weight | 8% |
| Single name min weight | 0% (long-only) |
| Emerging markets combined max | 20% |
| Fixed income combined max | 50% |
| Cash (BIL/SHV) max | 30% |
| Portfolio beta to AOA | 0.6 – 1.3 |

### 5c. Volatility Targeting
Scale the entire portfolio to target **12% annualized volatility**:

```
scale = vol_target / ewma_portfolio_vol_21d
final_weights = raw_weights × min(scale, 1.2)   # cap leverage at 1.2× for long-only
```

Since the portfolio is long-only, leverage is only applied when realized vol
is meaningfully below target. In practice, the cap rarely binds.

---

## 6. Benchmark Comparison vs AOA

AOA holds approximately: 60% global equities, 40% bonds (static, rebalanced quarterly).

| Characteristic | AOA | This Strategy |
|---------------|-----|--------------|
| Equity allocation | ~60% fixed | 20–80% dynamic |
| Bond allocation | ~40% fixed | 10–60% dynamic |
| Duration management | Passive | Active regime-steered |
| Momentum tilt | None | Top-quintile overweight |
| Regime awareness | None | CJM bear probability |
| Rebalance | Quarterly | Monthly |
| Expected excess return | — | +2 to +4% annualized |
| Expected tracking error | — | 8–14% |
| Expected Info Ratio | — | 0.25–0.40 |

---

## 7. Backtesting Framework

### 7a. Walk-Forward Protocol
- **Training window:** 5 years rolling
- **Test window:** 1 year out-of-sample
- **Embargo:** 21 trading days between train and test (prevent leakage)
- **Start:** 2010-01-01 (sufficient history for all Tier-1 tickers)
- **End:** 2024-12-31

### 7b. Execution Assumptions
| Item | Assumption |
|------|-----------|
| Trade execution | Next-day open (T+1) |
| Transaction cost | 5 bps one-way (liquid ETFs) |
| Market impact | 2 bps for names > $2B ADV; 5 bps otherwise |
| Rebalance date | Last business day of each month |
| Dividends | Reinvested (use adj_close) |

### 7c. Performance Metrics
- Annualized return (CAGR)
- Annualized Sharpe ratio (risk-free = T-bill from BIL/SHV series)
- Maximum drawdown
- Calmar ratio (CAGR / Max DD)
- Excess return vs AOA
- Information ratio
- Deflated Sharpe Ratio (López de Prado) — adjusts for multiple testing
- Hit rate of monthly regime calls (bull/bear vs realized)

---

## 8. Implementation Roadmap

```
Phase 1 — Data Preparation
  ├── Load all CSVs from Data/raw/Macro and Prices/
  ├── Compute daily adj_close returns for all tickers
  ├── Align on common trading calendar
  └── Compute SPY-based features for CJM

Phase 2 — Jump Model Training
  ├── Train CJM on SPY features (walk-forward)
  ├── Generate p_bear time series via predict_online()
  └── Validate regime labels against known bear markets (2008, 2020, 2022)

Phase 3 — Signal Construction
  ├── Compute 12-1 momentum ranks for all Tier-1 equities
  ├── Compute 60-day realized vol for inverse-vol weighting
  └── Compute EWMA portfolio vol for vol targeting

Phase 4 — Portfolio Construction
  ├── Apply regime-dependent sleeve weights
  ├── Apply position limits and normalize
  └── Apply vol targeting scalar

Phase 5 — Backtesting
  ├── Run walk-forward backtest 2010–2024
  ├── Compute all performance metrics
  └── Compare to AOA buy-and-hold

Phase 6 — Analysis & Iteration
  ├── Regime analysis: performance by regime state
  ├── Attribution: sleeve-level contribution
  └── Parameter sensitivity (λ for CJM, momentum lookback, vol target)
```

---

## 9. Key Files Reference

| File | Purpose |
|------|---------|
| `Data/raw/Macro and Prices/AOA.csv` | Benchmark |
| `Data/raw/Macro and Prices/SPY.csv` | CJM regime features |
| `Data/raw/Macro and Prices/VIX_History.csv` | Supplemental regime feature |
| `Data/raw/Macro and Prices/^IDCOTSTR.csv` | Credit spread regime feature |
| `Data/raw/Macro and Prices/Treasuries_Historical Data.csv` | Yield curve for duration rotation |
| `Data/raw/Macro and Prices/BIL.csv` | Risk-free rate proxy |
| `jumpmodels/` | CJM model implementation |
| `examples/nasdaq/example.ipynb` | CJM usage reference |
| `docs/DATA_REQUEST.md` | Original data specification |

---

*End of Strategy Framework v2.0*
