import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
# Make stdout/stderr UTF-8 so status glyphs print on any console (Windows uses cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Script-relative paths so the script works from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load strategy returns
returns = pd.read_csv(os.path.join(HERE, 'portfolio_returns.csv'))
returns['date'] = pd.to_datetime(returns['date'])
returns = returns.sort_values('date').reset_index(drop=True)

# Build a REAL equal-weighted benchmark from df_2010.csv instead of the
# previous hardcoded `market_annual_return = 0.1322` that produced a smooth
# deterministic exponential with zero volatility and zero drawdown.
# This matches the benchmark definition used inside momentum_ml_framework.py
# (Section II.B of the paper: equal-weighted average of the 100-stock universe).
prices = pd.read_csv(os.path.join(HERE, 'df_2010.csv'))
prices['date'] = pd.to_datetime(prices['date'])
prices = prices.rename(columns={'PX_LAST': 'Close', 'ticker': 'asset'})

# Daily equal-weighted returns: per-date mean of asset pct_changes
close_panel = prices.pivot(index='date', columns='asset', values='Close').sort_index()
asset_daily_rets = close_panel.pct_change()
benchmark_daily = asset_daily_rets.mean(axis=1).dropna()

# Align benchmark to the strategy's date range
strategy_dates = returns['date']
benchmark_aligned = benchmark_daily.reindex(strategy_dates).fillna(0.0).values

# Strategy metrics
strategy_returns = returns['return'].values
strategy_cum = (1 + strategy_returns).cumprod()
strategy_total_return = strategy_cum[-1] - 1
n_days = len(strategy_returns)
strategy_annual_return = (1 + strategy_total_return) ** (252 / n_days) - 1

# Benchmark metrics (REAL — from data, not hardcoded)
benchmark_cum = (1 + benchmark_aligned).cumprod()
benchmark_total_return = benchmark_cum[-1] - 1
benchmark_annual_return = (1 + benchmark_total_return) ** (252 / n_days) - 1
n_years = n_days / 252.0

print(f"Strategy : total {strategy_total_return*100:.2f}%  ann {strategy_annual_return*100:.2f}%")
print(f"Benchmark: total {benchmark_total_return*100:.2f}%  ann {benchmark_annual_return*100:.2f}%  (equal-weighted, {n_years:.1f}y)")

print("Creating charts...")

# Chart 1: Cumulative performance
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(strategy_cum, linewidth=2.5, label='ML Momentum Strategy', color='#2E86AB')
ax.plot(benchmark_cum, linewidth=2.5, label='Equal-Weighted Benchmark', color='#A23B72', linestyle='--')
ax.set_title('Cumulative Performance', fontsize=16, fontweight='bold')
ax.set_xlabel('Trading Days', fontsize=12)
ax.set_ylabel('Portfolio Value ($1 initial)', fontsize=12)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'Strategy: ${strategy_cum[-1]:.2f} ({strategy_total_return*100:.1f}%)',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.text(0.05, 0.88, f'Benchmark: ${benchmark_cum[-1]:.2f} ({benchmark_total_return*100:.1f}%)',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
plt.savefig(os.path.join(FIG_DIR, 'CHART_1_Performance.png'), dpi=200)
plt.close()
print("OK CHART_1_Performance.png")

# Chart 2: Drawdown
running_max = pd.Series(strategy_cum).expanding().max()
drawdown = (strategy_cum - running_max.values) / running_max.values

bench_running_max = pd.Series(benchmark_cum).expanding().max()
bench_drawdown = (benchmark_cum - bench_running_max.values) / bench_running_max.values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax1.plot(strategy_cum, linewidth=2, color='#2E86AB', label='Strategy')
ax1.plot(benchmark_cum, linewidth=2, color='#A23B72', linestyle='--', label='Benchmark')
ax1.fill_between(range(len(strategy_cum)), strategy_cum, running_max.values,
                  where=strategy_cum < running_max.values, alpha=0.3, color='red')
ax1.set_title('Portfolio Value with Drawdown Periods', fontsize=14, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.6, color='#E63946', label=f'Strategy (Max: {drawdown.min()*100:.2f}%)')
ax2.plot(bench_drawdown * 100, color='#A23B72', linestyle='--', linewidth=1.5, label=f'Benchmark (Max: {bench_drawdown.min()*100:.2f}%)')
ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Trading Days', fontsize=12)
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.legend()
plt.savefig(os.path.join(FIG_DIR, 'CHART_2_Drawdown.png'), dpi=200)
plt.close()
print("OK CHART_2_Drawdown.png")

# Chart 3: Annual returns (strategy + benchmark side by side)
returns['year'] = returns['date'].dt.year
annual_returns = returns.groupby('year')['return'].apply(lambda x: (1 + x).prod() - 1) * 100

bench_df = pd.DataFrame({'date': strategy_dates, 'return': benchmark_aligned})
bench_df['year'] = bench_df['date'].dt.year
bench_annual = bench_df.groupby('year')['return'].apply(lambda x: (1 + x).prod() - 1) * 100

years = annual_returns.index.tolist()
x = np.arange(len(years))
width = 0.4

fig, ax = plt.subplots(figsize=(13, 7))
ax.bar(x - width/2, annual_returns.values, width, label='Strategy', color='#2E86AB', alpha=0.85, edgecolor='black')
ax.bar(x + width/2, bench_annual.reindex(years).values, width, label='Benchmark', color='#A23B72', alpha=0.85, edgecolor='black')
ax.set_title('Annual Returns by Year', fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Annual Return (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'CHART_3_Annual_Returns.png'), dpi=200)
plt.close()
print("OK CHART_3_Annual_Returns.png")

# Chart 4: Metrics summary
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

volatility = strategy_returns.std() * np.sqrt(252) * 100
bench_volatility = benchmark_aligned.std() * np.sqrt(252) * 100
# Match the daily-Sharpe formula used in momentum_ml_framework.py:
#   Sharpe = sqrt(252) * mean(daily - rf/252) / std(daily)
rf = 0.02
sharpe = np.sqrt(252) * (strategy_returns - rf/252).mean() / strategy_returns.std()
bench_sharpe = np.sqrt(252) * (benchmark_aligned - rf/252).mean() / benchmark_aligned.std() if benchmark_aligned.std() > 0 else 0
win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100
alpha = (strategy_annual_return - benchmark_annual_return) * 100

metrics_data = [
    ['METRIC', 'STRATEGY', 'BENCHMARK'],
    ['', '', ''],
    [f'Total Return ({n_years:.1f} yrs)', f'{strategy_total_return*100:.2f}%', f'{benchmark_total_return*100:.2f}%'],
    ['Annualized Return', f'{strategy_annual_return*100:.2f}%', f'{benchmark_annual_return*100:.2f}%'],
    ['Alpha (Annual)', f'{alpha:.2f}%', '-'],
    ['', '', ''],
    ['Volatility (Annual)', f'{volatility:.2f}%', f'{bench_volatility:.2f}%'],
    ['Sharpe Ratio (rf=2%)', f'{sharpe:.3f}', f'{bench_sharpe:.3f}'],
    ['Max Drawdown', f'{drawdown.min()*100:.2f}%', f'{bench_drawdown.min()*100:.2f}%'],
    ['', '', ''],
    ['Win Rate (daily)', f'{win_rate:.2f}%', '-'],
    ['Number of Days', f'{len(returns)}', '-'],
    ['Period', f'{returns["date"].iloc[0].date()} to {returns["date"].iloc[-1].date()}', '-'],
]

table = ax.table(cellText=metrics_data, cellLoc='center', loc='center', colWidths=[0.45, 0.275, 0.275])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

for i in range(3):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white', size=13)

table[(4, 1)].set_facecolor('#C8E6C9')
table[(4, 1)].set_text_props(weight='bold')

plt.title('Strategy Performance Summary', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(os.path.join(FIG_DIR, 'CHART_4_Metrics_Table.png'), dpi=200)
plt.close()
print("OK CHART_4_Metrics_Table.png")

# Chart 5: Distribution of returns
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(strategy_returns * 100, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
axes[0].axvline(x=strategy_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {strategy_returns.mean()*100:.3f}%')
axes[0].set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Daily Return (%)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

box_data = [strategy_returns * 100]
# matplotlib >=3.9 deprecates `labels=` in boxplot; use `tick_labels=` instead.
bp = axes[1].boxplot(box_data, tick_labels=['Strategy'], patch_artist=True, showmeans=True, vert=True)
bp['boxes'][0].set_facecolor('#2E86AB')
axes[1].set_title('Returns Distribution (Box Plot)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Daily Return (%)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)

plt.savefig(os.path.join(FIG_DIR, 'CHART_5_Returns_Distribution.png'), dpi=200)
plt.close()
print("OK CHART_5_Returns_Distribution.png")

print("\n" + "="*70)
print("ALL CHARTS SUCCESSFULLY CREATED")
print("="*70)
