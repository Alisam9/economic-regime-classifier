# %%
# =============================================================================
# US ECONOMIC REGIME CLASSIFIER — v2
# =============================================================================
# Author  : Alireza Samanian
# Purpose : Classify the US business cycle into four regimes using a
#           5-indicator composite score with rolling Z-score normalization.
#           Outputs multi-asset performance analysis, regime momentum,
#           drawdown/CVaR risk metrics, and a transition probability matrix.
#
# Indicators (sourced via FRED / OpenBB):
#   1. 10Y-2Y Treasury Spread       — yield curve shape
#   2. 10Y Breakeven Inflation       — inflation expectations (replaces 10Y-3M)
#   3. Moody's Baa-10Y Credit Spread — credit market risk premium
#   4. Initial Jobless Claims (YoY)  — labor market conditions
#   5. Industrial Production (YoY)   — real economic activity
# =============================================================================


# %% ------------------------------------------------------------------------
# CELL 1 — IMPORTS
# ---------------------------------------------------------------------------

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from openbb import obb

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
})

print("All packages loaded successfully.")


# %% ------------------------------------------------------------------------
# CELL 2 — CREDENTIALS
# ---------------------------------------------------------------------------
# Load FRED API key from environment variable.
# Set FRED_API_KEY in your shell or a .env file before running.
# Free key available at: https://fred.stlouisfed.org/docs/api/api_key.html

import os
from dotenv import load_dotenv
load_dotenv()

obb.user.credentials.fred_api_key = os.getenv("FRED_API_KEY")


# %% ------------------------------------------------------------------------
# CELL 3 — DATA COLLECTION: REGIME INDICATORS
# ---------------------------------------------------------------------------
# T10YIE (breakeven inflation) replaces T10Y3M to avoid yield curve redundancy
# and to capture the inflation dimension independently.
# Sign is inverted in Cell 5: rising inflation → tightening risk → negative regime signal.

INDICATORS = {
    "spread_10y2y":        "T10Y2Y",    # 10Y-2Y Treasury spread
    "breakeven_inflation": "T10YIE",    # 10Y TIPS breakeven inflation
    "credit_spread":       "BAA10Y",    # Moody's Baa minus 10Y Treasury
    "jobless_claims":      "ICSA",      # Weekly initial unemployment claims
    "indpro":              "INDPRO",    # Industrial Production Index
}

START_DATE = "1990-01-01"


def pull_fred_series(ticker, start_date, max_retries=3, sleep_sec=3):
    """Pull a FRED series via OpenBB with retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            result = obb.economy.fred_series(symbol=ticker, start_date=start_date)
            df     = result.to_df()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed — retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
            else:
                raise e


raw_data = {}
for name, ticker in INDICATORS.items():
    print(f"Pulling {name} ({ticker})...")
    raw_data[name] = pull_fred_series(ticker, START_DATE)
    print(f"  → {raw_data[name].shape[0]:,} rows | "
          f"{raw_data[name].index[0].date()} to {raw_data[name].index[-1].date()}")
    time.sleep(2)

print("\nAll indicators loaded.")


# %% ------------------------------------------------------------------------
# CELL 4 — DATA PROCESSING: ALIGN TO MONTHLY FREQUENCY
# ---------------------------------------------------------------------------
# Daily/weekly series resampled to month-start via monthly mean.
# INDPRO and jobless_claims converted to YoY % change to remove trend.

monthly = {}

for name in ["spread_10y2y", "breakeven_inflation", "credit_spread", "jobless_claims"]:
    monthly[name] = raw_data[name].resample("MS").mean()

monthly["indpro"] = raw_data["indpro"].copy()
monthly["indpro"].index = pd.to_datetime(monthly["indpro"].index)

df = pd.concat(monthly.values(), axis=1)
df.columns = list(monthly.keys())

df = df.ffill()

df["indpro"]         = df["indpro"].pct_change(12) * 100
df["jobless_claims"] = df["jobless_claims"].pct_change(12) * 100

df = df.dropna()

print(f"Processed DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
print(df.describe().round(2))


# %% ------------------------------------------------------------------------
# CELL 5 — ROLLING Z-SCORE NORMALIZATION
# ---------------------------------------------------------------------------
# Rolling window (60M) eliminates look-ahead bias present in full-sample normalization.
# min_periods=36 ensures at least 3 years of history before producing a signal.
#
# Sign conventions — all indicators normalized so HIGH = healthy, LOW = stressed:
#   credit_spread, jobless_claims, breakeven_inflation → signs inverted.

ROLLING_WINDOW = 60
MIN_PERIODS    = 36


def rolling_zscore(series, window=ROLLING_WINDOW, min_periods=MIN_PERIODS):
    """Z-score a series using a rolling historical window (no look-ahead bias)."""
    roll_mean = series.rolling(window, min_periods=min_periods).mean()
    roll_std  = series.rolling(window, min_periods=min_periods).std()
    return (series - roll_mean) / roll_std


df_norm = df.apply(rolling_zscore)

df_norm["credit_spread"]       = -df_norm["credit_spread"]
df_norm["jobless_claims"]      = -df_norm["jobless_claims"]
df_norm["breakeven_inflation"] = -df_norm["breakeven_inflation"]

df_norm = df_norm.dropna()

print(f"Normalized data: {df_norm.shape[0]} rows")
print(f"Date range: {df_norm.index[0].date()} → {df_norm.index[-1].date()}")
print(df_norm.describe().round(2))


# %% ------------------------------------------------------------------------
# CELL 6 — COMPOSITE SCORE & REGIME CLASSIFICATION
# ---------------------------------------------------------------------------
# Equal weighting (1/N) per DeMiguel, Garlappi & Uppal (2009): naive diversification
# consistently outperforms optimized weights out-of-sample due to estimation error.
#
# Regime thresholds (Z-score):
#   > +0.5  → Expansion
#    0–0.5  → Late Cycle
#   -0.5–0  → Slowdown
#   < -0.5  → Contraction
#
# Composite smoothed over 3 months before classification to reduce monthly noise.

WEIGHTS = {
    "spread_10y2y":        0.20,
    "breakeven_inflation": 0.20,
    "credit_spread":       0.20,
    "jobless_claims":      0.20,
    "indpro":              0.20,
}

THRESHOLDS = {
    "expansion":  0.5,
    "late_cycle": 0.0,
    "slowdown":  -0.5,
}

REGIME_ORDER  = ["Expansion", "Late Cycle", "Slowdown", "Contraction"]
REGIME_COLORS = {
    "Expansion":   "#2ecc71",
    "Late Cycle":  "#f39c12",
    "Slowdown":    "#e67e22",
    "Contraction": "#e74c3c",
}

df_norm["composite"] = sum(
    df_norm[col] * w for col, w in WEIGHTS.items()
)

df_norm["composite_smooth"] = df_norm["composite"].rolling(3).mean()
df_norm = df_norm.dropna(subset=["composite_smooth"])


def classify_regime(score):
    """Map a composite Z-score to one of four business cycle regimes."""
    if score > THRESHOLDS["expansion"]:
        return "Expansion"
    elif score > THRESHOLDS["late_cycle"]:
        return "Late Cycle"
    elif score > THRESHOLDS["slowdown"]:
        return "Slowdown"
    else:
        return "Contraction"


df_norm["regime"] = df_norm["composite_smooth"].apply(classify_regime)

print(f"Current composite (smoothed) : {df_norm['composite_smooth'].iloc[-1]:.3f}")
print(f"Current regime               : {df_norm['regime'].iloc[-1]}")
print(f"\nHistorical regime distribution ({df_norm.shape[0]} months):")
dist = df_norm["regime"].value_counts()
for r in REGIME_ORDER:
    count = dist.get(r, 0)
    pct   = count / df_norm.shape[0] * 100
    print(f"  {r:<15} {count:>3} months  ({pct:.1f}%)")


# %% ------------------------------------------------------------------------
# CELL 7 — REGIME MOMENTUM
# ---------------------------------------------------------------------------
# 3-month change in smoothed composite — captures directional trend within a regime.
# Positive = conditions improving; negative = deteriorating.

df_norm["momentum"] = df_norm["composite_smooth"].diff(3)

current_momentum = df_norm["momentum"].iloc[-1]
momentum_label   = "Improving ↑" if current_momentum > 0 else "Deteriorating ↓"

print(f"3-month composite momentum: {current_momentum:+.3f}  ({momentum_label})")


# %% ------------------------------------------------------------------------
# CELL 8 — NBER RECESSION DATA & FINAL MERGE
# ---------------------------------------------------------------------------
# USREC: NBER-dated binary recession indicator (1 = recession).
# Used for chart overlay and out-of-sample validation only — not a model input.

recession_raw = obb.economy.fred_series(symbol="USREC", start_date=START_DATE)
df_rec = recession_raw.to_df()
df_rec.index = pd.to_datetime(df_rec.index)
df_rec = df_rec.resample("MS").last()
df_rec.columns = ["recession"]

df_final = df_norm[["composite_smooth", "regime", "momentum"]].join(
    df_rec, how="left"
)
df_final["recession"] = df_final["recession"].fillna(0)

print(f"Final dataset: {df_final.shape[0]} rows")
print(f"Recession months in sample: {int(df_final['recession'].sum())}")


# %% ------------------------------------------------------------------------
# CELL 9 — MAIN CHART: COMPOSITE SCORE + REGIME STRIP
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(16, 8),
    gridspec_kw={"height_ratios": [4, 1]},
    sharex=True
)

# Top panel: composite score with NBER shading and regime thresholds
ax1.fill_between(
    df_final.index, -4, 2,
    where=(df_final["recession"] == 1.0),
    color="gray", alpha=0.15, label="NBER Recession"
)
ax1.plot(
    df_final.index, df_final["composite_smooth"],
    color="steelblue", linewidth=1.8,
    label="Composite Score (3M Smoothed)", zorder=3
)
ax1.axhline( 0.5, color=REGIME_COLORS["Expansion"],  linestyle=":", linewidth=1.0, alpha=0.8)
ax1.axhline( 0.0, color="black",                      linestyle="--", linewidth=0.8, alpha=0.4)
ax1.axhline(-0.5, color=REGIME_COLORS["Contraction"], linestyle=":", linewidth=1.0, alpha=0.8)

last_date   = df_final.index[-1]
last_value  = df_final["composite_smooth"].iloc[-1]
last_regime = df_final["regime"].iloc[-1]

ax1.scatter(last_date, last_value, color="darkblue", s=70, zorder=5)
ax1.annotate(
    f"Now: {last_value:.2f}  |  {last_regime}  |  {momentum_label}",
    xy=(last_date, last_value),
    xytext=(-170, -35), textcoords="offset points",
    fontsize=9, color="darkblue",
    arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.2)
)

ax1.set_title(
    "US Economic Regime Composite Score  —  Rolling Z-Score  |  5 Indicators",
    fontsize=14, fontweight="bold", pad=15
)
ax1.set_ylabel("Composite Z-Score", fontsize=11)
ax1.set_ylim(-4, 2)
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Bottom panel: regime classification strip
for regime, color in REGIME_COLORS.items():
    ax2.fill_between(
        df_final.index, 0, 1,
        where=(df_final["regime"] == regime),
        color=color, alpha=0.85, label=regime
    )
ax2.set_yticks([])
ax2.set_ylabel("Regime", fontsize=10)
ax2.legend(loc="upper left", fontsize=8, ncol=4, bbox_to_anchor=(0, 1.4))
ax2.grid(False)

plt.tight_layout()
plt.show()


# %% ------------------------------------------------------------------------
# CELL 10 — MULTI-ASSET DATA COLLECTION
# ---------------------------------------------------------------------------
# S&P 500  : yfinance (^GSPC), monthly close, full history available.
#
# 10Y Bond : No free total return index available for this period.
#            Approximated via modified duration: Return ≈ -8 × Δyield (%)
#            Duration = 8 (conservative for on-the-run 10Y). Source: GS10 (FRED).
#
# Gold     : GLD ETF (SPDR, ~0.40% annual expense ratio) via yfinance.
#            FRED's GOLDAMGBD228NLBM is unavailable through the OpenBB free provider.
#            GLD launched Nov 2004; analysis period begins 2006 to allow warmup.

print("Pulling S&P 500...")
sp500_raw = yf.download("^GSPC", start="1991-01-01", interval="1mo", auto_adjust=True)
df_sp500  = sp500_raw[["Close"]].copy()
df_sp500.index = df_sp500.index.to_period("M").to_timestamp()
df_sp500.columns = ["sp500"]
df_sp500["sp500_return"] = df_sp500["sp500"].pct_change() * 100
print(f"  → {df_sp500.shape[0]} rows | {df_sp500.index[0].date()} to {df_sp500.index[-1].date()}")

print("Pulling 10Y Treasury yield (GS10)...")
gs10_raw  = obb.economy.fred_series(symbol="GS10", start_date=START_DATE)
df_gs10   = gs10_raw.to_df()
df_gs10.index = pd.to_datetime(df_gs10.index)
df_gs10   = df_gs10.resample("MS").last()
df_gs10.columns = ["gs10"]
df_gs10["bond_return"] = -8 * df_gs10["gs10"].diff()
print(f"  → {df_gs10.shape[0]} rows | {df_gs10.index[0].date()} to {df_gs10.index[-1].date()}")

print("Pulling Gold (GLD ETF)...")
gold_raw = yf.download("GLD", start="2004-01-01", interval="1mo", auto_adjust=True)
df_gold  = gold_raw[["Close"]].copy()
df_gold.index = df_gold.index.to_period("M").to_timestamp()
df_gold.columns = ["gold"]
df_gold["gold_return"] = df_gold["gold"].pct_change() * 100
print(f"  → {df_gold.shape[0]} rows | {df_gold.index[0].date()} to {df_gold.index[-1].date()}")


# %% ------------------------------------------------------------------------
# CELL 11 — MERGE ASSETS WITH REGIME DATA
# ---------------------------------------------------------------------------

df_assets = df_final[["regime"]].join(
    [df_sp500["sp500_return"], df_gs10["bond_return"], df_gold["gold_return"]],
    how="left"
).dropna()

print(f"Combined asset + regime dataset: {df_assets.shape}")
print(f"Date range: {df_assets.index[0].date()} → {df_assets.index[-1].date()}")
print(df_assets.tail(6))


# %% ------------------------------------------------------------------------
# CELL 12 — PERFORMANCE STATISTICS BY REGIME
# ---------------------------------------------------------------------------
# Metrics computed per asset × per regime:
#   Avg Return, Volatility, Hit Rate, Return/Risk (simplified Sharpe),
#   Max Drawdown, CVaR 5% (Expected Shortfall at 5th percentile).

ASSETS = {
    "sp500_return": "S&P 500",
    "bond_return":  "10Y Treasury (Approx)",
    "gold_return":  "Gold",
}


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown (%) within a return series."""
    cumulative  = (1 + returns / 100).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max * 100
    return drawdown.min()


def cvar_5pct(returns):
    """Conditional Value at Risk at 5% — mean return in the worst 5% of months."""
    threshold = returns.quantile(0.05)
    tail      = returns[returns <= threshold]
    return tail.mean() if len(tail) > 0 else np.nan


all_stats = {}

for col, label in ASSETS.items():
    grouped = df_assets.groupby("regime")[col]
    stats   = pd.DataFrame({
        "Avg Return (%)":   grouped.mean(),
        "Volatility (%)":   grouped.std(),
        "Hit Rate (%)":     grouped.apply(lambda x: (x > 0).mean() * 100),
        "Return/Risk":      grouped.mean() / grouped.std(),
        "Max Drawdown (%)": grouped.apply(max_drawdown),
        "CVaR 5% (%)":      grouped.apply(cvar_5pct),
        "Months":           grouped.count(),
    }).round(2).loc[REGIME_ORDER]

    all_stats[label] = stats

    print(f"\n{'='*65}")
    print(f"  {label}  —  Performance by Regime")
    print(f"{'='*65}")
    print(stats.to_string())

print("\n\nNote: Max Drawdown and CVaR are negative by definition.")
print("Regime samples with < 50 months should be interpreted cautiously.")


# %% ------------------------------------------------------------------------
# CELL 13 — ASSET ALLOCATION CHART (3 × 3 GRID)
# ---------------------------------------------------------------------------
# Rows: S&P 500 | 10Y Treasury | Gold
# Columns: Avg Return | Volatility | CVaR 5%

PLOT_METRICS = [
    ("Avg Return (%)",  "Avg Monthly Return (%)"),
    ("Volatility (%)",  "Monthly Volatility (Std Dev)"),
    ("CVaR 5% (%)",     "CVaR 5%  (Avg Worst Month)"),
]

bar_colors = [REGIME_COLORS[r] for r in REGIME_ORDER]

fig, axes = plt.subplots(3, 3, figsize=(18, 13))

for row_idx, (asset_label, stats_df) in enumerate(all_stats.items()):
    for col_idx, (metric_col, metric_title) in enumerate(PLOT_METRICS):
        ax     = axes[row_idx][col_idx]
        values = stats_df[metric_col]

        bars = ax.bar(
            REGIME_ORDER, values,
            color=bar_colors, alpha=0.85,
            edgecolor="white", linewidth=1.2
        )

        for bar, val in zip(bars, values):
            if val >= 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(values.max()) * 0.03,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold"
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - abs(values.min()) * 0.05,
                    f"{val:.2f}",
                    ha="center", va="top",
                    fontsize=9, fontweight="bold"
                )

        ax.set_title(
            f"{asset_label}\n{metric_title}",
            fontsize=10, fontweight="bold"
        )
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)

plt.suptitle(
    "Multi-Asset Performance by Economic Regime  —  Rolling Z-Score Model",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.show()


# %% ------------------------------------------------------------------------
# CELL 14 — REGIME TRANSITION PROBABILITY MATRIX
# ---------------------------------------------------------------------------
# Markov chain transition matrix: historical frequency of month-to-month
# regime changes. Rows = current regime; columns = next month's regime.
# Each row sums to 1.0. High diagonal values indicate regime persistence.

regimes_list = df_final["regime"].tolist()

transition_counts = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER)

for i in range(len(regimes_list) - 1):
    from_r = regimes_list[i]
    to_r   = regimes_list[i + 1]
    if from_r in REGIME_ORDER and to_r in REGIME_ORDER:
        transition_counts.loc[from_r, to_r] += 1

row_sums = transition_counts.sum(axis=1)
transition_prob = transition_counts.div(row_sums, axis=0).round(3)

print("Regime Transition Probability Matrix")
print("Row = current regime  |  Column = next month's regime\n")
print(transition_prob.map(lambda x: f"{x:.1%}").to_string())

fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(
    transition_prob,
    annot=True,
    fmt=".1%",
    cmap="RdYlGn",
    vmin=0, vmax=1,
    linewidths=0.8,
    linecolor="white",
    ax=ax,
    cbar_kws={"label": "Transition Probability", "shrink": 0.8},
    annot_kws={"size": 13, "weight": "bold"}
)

ax.set_title(
    "Regime Transition Probability Matrix\n"
    "Row = Current Regime  |  Column = Next Month's Regime",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Next Month Regime", fontsize=11)
ax.set_ylabel("Current Regime", fontsize=11)
ax.tick_params(axis="both", labelsize=11)

current_regime = df_final["regime"].iloc[-1]
current_idx    = REGIME_ORDER.index(current_regime)
ax.add_patch(plt.Rectangle(
    (0, current_idx), len(REGIME_ORDER), 1,
    fill=False, edgecolor="darkblue", lw=3.5,
    label=f"Current regime: {current_regime}"
))
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.show()


# %% ------------------------------------------------------------------------
# CELL 15 — CURRENT SIGNAL DASHBOARD
# ---------------------------------------------------------------------------

DIVIDER = "=" * 60

print(f"\n{DIVIDER}")
print(f"  US ECONOMIC REGIME DASHBOARD")
print(f"  As of: {df_final.index[-1].strftime('%B %Y')}")
print(DIVIDER)
print(f"  Composite Score (Smoothed) : {df_final['composite_smooth'].iloc[-1]:+.3f}")
print(f"  Current Regime             : {df_final['regime'].iloc[-1]}")
print(f"  3-Month Momentum           : {df_final['momentum'].iloc[-1]:+.3f}  ({momentum_label})")
print("-" * 60)

INDICATOR_LABELS = {
    "spread_10y2y":        "Yield Curve (10Y-2Y)",
    "breakeven_inflation": "Breakeven Inflation  (inverted)",
    "credit_spread":       "Credit Spread        (inverted)",
    "jobless_claims":      "Jobless Claims       (inverted)",
    "indpro":              "Industrial Production",
}

print("  Individual Indicator Z-Scores:")
for col, label in INDICATOR_LABELS.items():
    val    = df_norm[col].iloc[-1]
    blocks = "█" * min(int(abs(val) * 4), 15)
    sign   = "+" if val > 0 else ""
    print(f"  {label:<38} {sign}{val:>6.2f}  {blocks}")

print("-" * 60)

print(f"\n  From '{current_regime}' — historical next-month probabilities:")
for to_r in REGIME_ORDER:
    prob   = transition_prob.loc[current_regime, to_r]
    blocks = "█" * int(prob * 25)
    arrow  = "◀" if to_r == current_regime else " "
    print(f"  → {to_r:<15} {prob:>6.1%}  {blocks} {arrow}")

print(DIVIDER)

latest_sp500 = df_sp500["sp500_return"].dropna().iloc[-1]
print(f"\n  Latest S&P 500 monthly return : {latest_sp500:+.2f}%")
print(f"  Latest 10Y yield              : {df_gs10['gs10'].iloc[-1]:.2f}%")
print(f"  Latest gold price             : ${df_gold['gold'].iloc[-1]:,.0f}/oz")
print(DIVIDER)
