"""
analyze_costs.py - the reality check.

Takes the backtest output (portfolio_returns.csv + portfolio_positions.csv) and asks
the two questions a gross backtest never answers:

  1. How much does the strategy actually trade, and what do transaction costs do to it?
  2. Is the "alpha" statistically distinguishable from market beta?

Run it AFTER the backtest (it needs portfolio_positions.csv and df_2010.csv):

    python momentum_ml_framework.py
    python analyze_costs.py

It prints a summary table and writes figures/CHART_7_Cost_Sensitivity.png.
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RF = 0.02            # annual risk-free rate (matches the backtest)
ANN = 252
COST_LEVELS_BPS = [5, 10, 20]   # one-way transaction cost, basis points


def _path(name):
    return os.path.join(HERE, name)


def load():
    r = pd.read_csv(_path("portfolio_returns.csv"), parse_dates=["date", "position_date"]).sort_values("date")
    g = r.set_index("date")["return"].astype(float)
    p = pd.read_csv(_path("portfolio_positions.csv"), parse_dates=["date"]).rename(columns={"0": "w"})
    W = p.pivot_table(index="date", columns="asset", values="w", fill_value=0.0).sort_index()
    px = pd.read_csv(_path("df_2010.csv"), parse_dates=["date"])
    P = px.pivot_table(index="date", columns="ticker", values="PX_LAST").sort_index()
    Rasset = P.pct_change()
    return r, g, W, Rasset


def turnover(W, Rasset):
    """Total fraction of the book traded each rebalance, drift-adjusted."""
    Wp = W.shift(1)
    rr = Rasset.reindex(index=W.index, columns=W.columns).fillna(0.0)
    Wd = Wp * (1 + rr)
    Wd = Wd.div(Wd.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    D = (W - Wd).abs().sum(axis=1)
    D.iloc[0] = np.nan
    return D


def metrics(x):
    x = pd.Series(x).dropna().values
    tot = np.prod(1 + x) - 1
    ann = (1 + tot) ** (ANN / len(x)) - 1
    vol = x.std(ddof=1) * np.sqrt(ANN)
    sharpe = np.sqrt(ANN) * (x - RF / ANN).mean() / x.std(ddof=1)
    cum = np.cumprod(1 + x)
    mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=mdd)


def jensen_alpha(strat, mkt):
    """Annualized alpha + HAC (Newey-West) t-stat and beta vs a market series."""
    import statsmodels.api as sm
    df = pd.concat([strat.rename("s"), mkt.rename("m")], axis=1).dropna()
    y = df["s"] - RF / ANN
    X = sm.add_constant(df["m"] - RF / ANN)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return res.params["const"] * ANN, res.tvalues["const"], res.params["m"], res.rsquared


def cost_chart(scenarios, bench_ann, turnover_x):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(scenarios.keys())
    vals = [scenarios[k] * 100 for k in labels]
    colors = ["#2a7fb8" if v >= 0 else "#c0392b" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(bench_ann * 100, color="#7b3294", linestyle="--", linewidth=1.5,
               label=f"Buy-and-hold benchmark ({bench_ann*100:.1f}%)")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net annualized return (%)")
    ax.set_title(f"Transaction costs destroy the edge\n~{turnover_x:.0f}x annual turnover; "
                 "cost shown is one-way, in basis points", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "CHART_7_Cost_Sensitivity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"OK {os.path.relpath(out, HERE)}")


def main():
    r, g, W, Rasset = load()
    D = turnover(W, Rasset)
    turnover_x = np.nanmean(D) / 2 * ANN

    # cost drag mapped to each return day via its position_date
    Dret = (r.merge(D.rename("D"), left_on="position_date", right_index=True, how="left")
            .set_index("date")["D"].reindex(g.index))

    scenarios = {"Gross": metrics(g)["ann"]}
    series = {0: g}
    for c_bps in COST_LEVELS_BPS:
        net = g - (c_bps / 1e4) * Dret
        scenarios[f"{c_bps} bps"] = metrics(net)["ann"]
        series[c_bps] = net

    bench = Rasset.reindex(index=g.index).mean(axis=1)
    bench_ann = metrics(bench)["ann"]

    print(f"\nAnnual one-way turnover: {turnover_x:.0f}x  "
          f"({np.nanmean(D)*100:.0f}% of the book traded per rebalance, daily)\n")
    print(f"{'scenario':<12}{'annRet':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>8}")
    for name, x in [("Gross", g)] + [(f"{c} bps", series[c]) for c in COST_LEVELS_BPS]:
        m = metrics(x)
        print(f"{name:<12}{m['ann']*100:>8.2f}%{m['vol']*100:>7.1f}%{m['sharpe']:>8.2f}{m['mdd']*100:>7.1f}%")
    print(f"{'Benchmark':<12}{bench_ann*100:>8.2f}%")

    a, t, b, r2 = jensen_alpha(g, bench)
    print(f"\nJensen alpha (gross) vs equal-weighted universe: "
          f"{a*100:+.2f}%/yr, t={t:.2f}, beta={b:.2f}, R^2={r2:.2f}")
    a, t, b, r2 = jensen_alpha(series[10], bench)
    print(f"Jensen alpha (net @10bps): {a*100:+.2f}%/yr, t={t:.2f}")

    cost_chart(scenarios, bench_ann, turnover_x)


if __name__ == "__main__":
    main()
