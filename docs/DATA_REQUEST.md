# Data Request: Tactical L/S ETF Strategy

**Project:** Institutional Tactical Long/Short ETF Strategy vs AOA Benchmark  
**Prepared by:** Quantitative Research  
**Date:** 2026-06-06  
**For:** Data Engineering / Data Acquisition Team

---

## 1. Overview

We are building a multi-sleeve tactical long/short ETF strategy across a universe of 42 liquid ETFs (pruned from a 67-name investable universe for institutional capacity). The strategy uses Statistical Jump Models for regime detection and requires clean, adjusted price history with supplemental data for portfolio construction and backtesting.

---

## 2. Primary Universe — 42 Active Tickers

These are the tickers to prioritize for full data download.

### 2a. US Equity Sectors (11 Vanguard ETFs)
| Ticker | Name |
|--------|------|
| VGT | Vanguard Information Technology ETF |
| VHT | Vanguard Health Care ETF |
| VFH | Vanguard Financials ETF |
| VCR | Vanguard Consumer Discretionary ETF |
| VDC | Vanguard Consumer Staples ETF |
| VIS | Vanguard Industrials ETF |
| VAW | Vanguard Materials ETF |
| VDE | Vanguard Energy ETF |
| VPU | Vanguard Utilities ETF |
| VOX | Vanguard Communication Services ETF |
| VNQ | Vanguard Real Estate ETF |

### 2b. International / Regional Equity ETFs (25 names)
| Ticker | Name |
|--------|------|
| EWJ | iShares MSCI Japan ETF |
| EWG | iShares MSCI Germany ETF |
| EWU | iShares MSCI United Kingdom ETF |
| EWL | iShares MSCI Switzerland ETF |
| EWQ | iShares MSCI France ETF |
| EWI | iShares MSCI Italy ETF |
| EWP | iShares MSCI Spain ETF |
| EWD | iShares MSCI Sweden ETF |
| EWN | iShares MSCI Netherlands ETF |
| EWO | iShares MSCI Austria ETF |
| EWK | iShares MSCI Belgium ETF |
| EWS | iShares MSCI Singapore ETF |
| EWA | iShares MSCI Australia ETF |
| EWC | iShares MSCI Canada ETF |
| EWZ | iShares MSCI Brazil ETF |
| EWW | iShares MSCI Mexico ETF |
| EWT | iShares MSCI Taiwan ETF |
| EWY | iShares MSCI South Korea ETF |
| EWM | iShares MSCI Malaysia ETF |
| EZA | iShares MSCI South Africa ETF |
| EWH | iShares MSCI Hong Kong ETF |
| THD | iShares MSCI Thailand ETF |
| TUR | iShares MSCI Turkey ETF |
| EIDO | iShares MSCI Indonesia ETF |
| EPHE | iShares MSCI Philippines ETF |

### 2c. Fixed Income / Treasury ETFs (6 names)
| Ticker | Name | Approx. Duration |
|--------|------|-----------------|
| EDV | Vanguard Extended Duration Treasury ETF | ~23.9 years |
| VGLT | Vanguard Long-Term Treasury ETF | ~16 years |
| VGIT | Vanguard Intermediate-Term Treasury ETF | ~5.4 years |
| VGSH | Vanguard Short-Term Treasury ETF | ~1.9 years |
| SHV | iShares Short Treasury Bond ETF | ~0.4 years |
| BIL | SPDR Bloomberg 1-3 Month T-Bill ETF | ~0.14 years |

### 2d. EM Fixed Income (1 name)
| Ticker | Name |
|--------|------|
| EMHY | iShares J.P. Morgan EM High Yield Bond ETF |

---

## 3. Benchmark

| Ticker | Name | Notes |
|--------|------|-------|
| AOA | iShares Core Aggressive Allocation ETF | Primary benchmark — download full history |

---

## 4. Secondary Universe — Metadata Only (remaining 25 names)

Download at minimum: ticker, full name, inception date, expense ratio, AUM (latest), average daily volume (latest 3 months). Price history is optional but welcome.

```
NGE, KWT, QAT, PAK, COLO, EGPT, FM, GXC, INDA, ECH, EPU, 
ARGT, ENZL, GREK, EDEN, EFNL, ENOR, EWW (already in primary), 
EWJE, VNM, ICOL, KORU, SMIN, FLMX, FLBR
```

*(Note: several of these are excluded from the active trading universe due to low AUM/ADV at institutional scale, but metadata is useful for universe review.)*

---

## 5. Data Fields Required

### 5a. Daily Price Data (OHLCV + Adjusted)
For every ticker in Sections 2 and 3:

| Field | Description |
|-------|-------------|
| `date` | Trading date (YYYY-MM-DD) |
| `open` | Unadjusted open price |
| `high` | Unadjusted high price |
| `low` | Unadjusted low price |
| `close` | Unadjusted close price |
| `volume` | Share volume traded |
| `adj_close` | Fully adjusted close (dividends + splits) |
| `dividends` | Cash dividend per share on ex-date |
| `splits` | Split ratio on split date (e.g., 2.0 = 2-for-1) |

### 5b. Derived / Computed Fields (nice to have, we can compute)
- Daily total return: `(adj_close[t] / adj_close[t-1]) - 1`
- 20-day and 60-day average daily volume (ADV) in USD: `close × volume` rolling mean

### 5c. Short Book Supplemental Data
For the 42 active tickers, we need short-selling feasibility data:

| Field | Description | Source Suggestions |
|-------|-------------|-------------------|
| `borrow_rate_ann` | Annualized stock borrow fee (%) | IB, S3 Partners, Markit |
| `borrow_availability` | Easy / Hard / Unavailable classification | IB Securities Lending |
| `short_interest` | Shares sold short / float (%) | FINRA, Bloomberg |

*Frequency: monthly snapshots are sufficient. Daily preferred if available.*

### 5d. Risk-Free Rate / Financing Cost
| Series | Description | Source |
|--------|-------------|--------|
| US 3-Month T-Bill yield | Daily, annualized (%) | FRED: `TB3MS` or `DTB3` |
| SOFR | Daily overnight rate (%) | FRED: `SOFR` |
| Fed Funds Effective Rate | Daily (%) | FRED: `DFF` |

*These are used for Sharpe ratio computation and long/short carry cost modeling.*

### 5e. Optional Macro / Regime Features
If bandwidth allows, the following are used as inputs to the Statistical Jump Model:

| Series | Description | Source | Frequency |
|--------|-------------|--------|-----------|
| VIX | CBOE Volatility Index | CBOE / Yahoo `^VIX` | Daily |
| MOVE Index | Treasury volatility index | ICE / Bloomberg | Daily |
| HY OAS | US High Yield Option-Adjusted Spread | FRED: `BAMLH0A0HYM2` | Daily |
| IG OAS | US Investment Grade OAS | FRED: `BAMLC0A0CM` | Daily |
| Yield Curve (10Y-2Y) | US Treasury slope | FRED: `T10Y2Y` | Daily |
| SPX | S&P 500 index level | Yahoo `^GSPC` | Daily |
| NASDAQ-100 | Nasdaq-100 index | Yahoo `^NDX` | Daily |

---

## 6. Date Range

| Parameter | Value |
|-----------|-------|
| **Start date** | Each ticker's **IPO / inception date** (get maximum available history) |
| **End date** | 2025-12-31 |
| **Minimum required start** | 2005-01-01 (earlier preferred) |
| **Frequency** | Daily (business days) |

> For backtesting walk-forward validation we need at least 15 years of history. Most of the iShares country ETFs launched 1996–2001, so full history is achievable.

---

## 7. Data Format & Delivery

### Preferred Format
- **Parquet** (preferred for performance) — one file per ticker, named `<TICKER>.parquet`  
  OR  
- **CSV** — one file per ticker, named `<TICKER>.csv`, UTF-8, comma-separated

### Directory Structure
```
data/
├── raw/
│   ├── prices/
│   │   ├── AOA.parquet
│   │   ├── VGT.parquet
│   │   ├── EWJ.parquet
│   │   └── ... (one file per ticker)
│   ├── macro/
│   │   ├── FRED_TB3MS.parquet
│   │   ├── FRED_DFF.parquet
│   │   ├── VIX.parquet
│   │   └── ...
│   └── short_data/
│       └── borrow_rates_monthly.parquet
└── metadata/
    └── universe_metadata.csv
```

### Column Naming Convention
Use **lowercase snake_case**: `date`, `open`, `high`, `low`, `close`, `volume`, `adj_close`, `dividends`, `splits`.

### Index
- `date` column should be parseable as `datetime64` (ISO 8601 format: `YYYY-MM-DD`)
- No duplicate dates per ticker
- Missing trading days (holidays) should be **absent** (not filled with NaN)

---

## 8. Data Sources (Suggested)

| Data Type | Suggested Sources |
|-----------|------------------|
| ETF OHLCV + adjusted prices | Yahoo Finance (`yfinance` library), Bloomberg, Refinitiv |
| Dividends & splits | `yfinance` (`.dividends`, `.splits`), Bloomberg corporate actions |
| AUM & expense ratios | ETF provider websites, ETFdb.com, Bloomberg |
| Short borrow rates | Interactive Brokers API, S3 Partners, Markit Securities Finance |
| FRED macro series | `pandas-datareader` FRED, or direct FRED API (`fredapi` library) |
| VIX | CBOE website, Yahoo Finance `^VIX` |

> **Note on `yfinance`:** Sufficient for initial prototyping. For production, cross-validate adjusted prices against Bloomberg or a paid data vendor, especially for older history and corporate action accuracy.

---

## 9. Acceptance Checklist

Before handing off the data, please verify:

- [ ] All 42 active tickers + AOA have price history starting no later than 2010-01-01
- [ ] Adjusted close prices reflect all dividends and splits (verify against a known distribution date)
- [ ] No duplicate rows per ticker
- [ ] No forward-looking data leakage (splits/dividends are recorded on their actual ex-date)
- [ ] FRED series `TB3MS` or `DTB3` available for full backtest window
- [ ] Short borrow data available for at least the 11 US sector ETFs and top 10 country ETFs by AUM
- [ ] All files loadable with `pd.read_parquet()` or `pd.read_csv(parse_dates=['date'])`
- [ ] Metadata file includes: ticker, full name, inception date, expense ratio, latest AUM, latest 3M ADV (USD)

---

## 10. Contact & Questions

Reach out to the quant research team with any questions on ticker scope, field definitions, or delivery format. For tickers not available through standard sources, flag them early so we can decide whether to substitute or drop from the universe.

---

*End of Data Request*
