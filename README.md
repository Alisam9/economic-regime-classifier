# US Economic Regime Classifier

A macro-driven business cycle model that classifies the current US economic regime using five independent indicators. Built in Python with FRED data via OpenBB.

---

## What It Does

The model combines five macro indicators into a single composite score, normalized using a rolling Z-score to eliminate look-ahead bias. The composite is mapped to one of four regimes — **Expansion**, **Late Cycle**, **Slowdown**, or **Contraction** — and paired with a momentum signal to capture directional drift within a regime.

The output includes historical regime classification back to the early 2000s, multi-asset performance statistics per regime (equities, bonds, gold), tail risk metrics (CVaR, max drawdown), and a Markov chain transition probability matrix for forward-looking regime probabilities.

---

## Indicators

| Indicator | FRED Series | Rationale |
|---|---|---|
| 10Y–2Y Treasury Spread | T10Y2Y | Yield curve shape — leading recession signal |
| 10Y Breakeven Inflation | T10YIE | Inflation expectations; replaces 10Y–3M to avoid yield curve redundancy |
| Moody's Baa–10Y Spread | BAA10Y | Credit risk premium — captures market stress independently of rates |
| Initial Jobless Claims (YoY) | ICSA | Labor market conditions with trend removed |
| Industrial Production (YoY) | INDPRO | Real economic activity with trend removed |

**Why these five?** Each indicator captures a structurally distinct dimension of the cycle: yield curve shape, inflation regime, credit conditions, labor market, and real output. Including both T10Y2Y and T10Y3M would double-count the yield curve signal — T10YIE was chosen over T10Y3M for this reason.

---

## Methodology

### Normalization
Each indicator is normalized using a **rolling 60-month Z-score** (minimum 36 months). Full-sample normalization would use future data to scale past observations — a look-ahead bias that makes backtests misleading. The rolling window ensures each month's score reflects only information available at that point in time.

### Composite Score
Indicators are combined using **equal weights (1/N)**. Per DeMiguel, Garlappi & Uppal (2009), naive diversification consistently outperforms statistically optimized weights out-of-sample due to estimation error in the covariance matrix. Optimization would fit historical noise rather than signal.

### Regime Classification
The smoothed composite (3-month moving average) is mapped to regimes using Z-score thresholds:

| Regime | Composite Score |
|---|---|
| Expansion | > +0.5 |
| Late Cycle | 0.0 to +0.5 |
| Slowdown | −0.5 to 0.0 |
| Contraction | < −0.5 |

### Momentum
A 3-month change in the smoothed composite captures directional trend within a regime. A Late Cycle reading at +0.3 and rising is a materially different signal than the same reading at +0.3 and falling toward Slowdown.

---

## Risk Metrics

Beyond average returns, each regime is characterized by **CVaR at 5%** (Conditional Value at Risk, also called Expected Shortfall) and **maximum drawdown**.

Standard deviation understates tail risk in financial returns, which are fat-tailed. VaR addresses this partially but only identifies the loss threshold — it says nothing about the magnitude of losses beyond that threshold. CVaR is the mean return in the worst 5% of months, making it a more complete tail risk measure and the standard in institutional risk management frameworks.

---

## Asset Coverage

| Asset | Source | Notes |
|---|---|---|
| S&P 500 | yfinance (^GSPC) | Monthly close, full history |
| 10Y Treasury | FRED GS10 + duration approximation | No free total return index available; return approximated as −8 × Δyield |
| Gold | yfinance (GLD ETF) | SPDR Gold Trust; ~0.40% annual expense ratio; history from 2004 |

---

## Key Findings (as of April 2026)

**Current reading:** Late Cycle | Composite +0.129 | Momentum −0.183 (Deteriorating)

**Transition probabilities from Late Cycle:** 88.4% probability of remaining Late Cycle next month; 8.4% probability of transitioning to Slowdown.

**Selected regime statistics:**

| Regime | S&P 500 Avg Monthly Return | S&P 500 Max Drawdown | Gold CVaR 5% |
|---|---|---|---|
| Expansion | +1.42% | −14.2% | −5.81% |
| Late Cycle | +0.89% | −19.3% | −6.12% |
| Slowdown | +0.21% | −28.7% | −5.94% |
| Contraction | −1.03% | −47.5% | −7.23% |

Notable: bonds returned −0.23% on average during Slowdown periods (dominated by the 2022 rate shock), illustrating the breakdown of the traditional 60/40 hedge. Gold was a Contraction hedge (+1.27% avg) but provided minimal protection in Slowdown (+0.25%).

---

## Limitations

- **Sample size**: Contraction months are underrepresented (~17 months post-2006 data start). Regime-level statistics should be interpreted cautiously given small samples.
- **Bond approximation**: The duration-based return estimate assumes constant 8-year modified duration. Actual total returns vary with coupon reinvestment and convexity.
- **T10YIE history**: Breakeven inflation data begins in 2003, reducing the full sample from 1990 to a post-2006 effective start after the rolling Z-score warmup period.
- **No out-of-sample validation**: Thresholds and weights were not optimized on return data, but the model has not been formally tested on a held-out period.
- **Regime persistence**: High diagonal values in the transition matrix (82–88%) reflect genuine persistence, but also the smoothing applied before classification.

---

## Setup

```bash
git clone https://github.com/Alisam9/economic-regime-classifier.git
cd economic-regime-classifier
pip install openbb pandas numpy matplotlib seaborn yfinance python-dotenv
cp .env.example .env
# Add your FRED API key to .env (free at fred.stlouisfed.org)
```

Run in VS Code with the Jupyter extension: open `regime_classifier_v2.py` and execute cells with `Shift+Enter`.

---

## Data Sources

- [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/) — all macro indicators via OpenBB
- [yfinance](https://github.com/ranaroussi/yfinance) — S&P 500 and GLD price data
- [NBER recession dating](https://www.nber.org/research/business-cycle-dating) — recession shading (USREC via FRED)
