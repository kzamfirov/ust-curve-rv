"""
Run the full study: PCA -> hedge test -> predictability test -> backtest -> robustness.
Outputs go to figures/ and results/.   Usage:  python run_all.py
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_yields
from src.pca import fit_pca
from src.fly import fly_series
from src.backtest import Params, run, summarise, monthly_weights, PCA_TENORS, half_life

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

STRESS = {
    "GFC (Sep 08-Mar 09)": ("2008-09-01", "2009-03-31"),
    "Taper tantrum (May-Sep 13)": ("2013-05-01", "2013-09-30"),
    "COVID dash for cash (Feb-Apr 20)": ("2020-02-20", "2020-04-30"),
    "Fed hiking cycle (2022)": ("2022-01-01", "2022-12-31"),
    "SVB / regional banks (Mar 23)": ("2023-03-01", "2023-03-31"),
    "Tariff shock (Apr 25)": ("2025-04-01", "2025-04-30"),
}

y = load_yields()
dy = y[PCA_TENORS].diff().mul(100).dropna()
out = {"sample": f"{y.index[0].date()} to {y.index[-1].date()}", "days": len(y)}

# 1. Full-sample PCA -------------------------------------------------------
L, ex = fit_pca(dy)
out["explained_variance"] = ex.round(4).to_dict()
L.round(4).to_csv(RES / "pca_loadings_full_sample.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
for k, c in enumerate(L.columns):
    ax[0].plot(L.index, L[c], marker="o", color=C[k], label=f"{c} ({ex[c]:.1%})")
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xscale("log"); ax[0].set_xticks(PCA_TENORS); ax[0].set_xticklabels([f"{t:g}y" for t in PCA_TENORS])
ax[0].set_title("PCA loadings: how each tenor moves per factor"); ax[0].set_xlabel("Maturity"); ax[0].legend()
ax[1].bar(range(1, 4), ex.values * 100, color=C[:3])
ax[1].set_xticks(range(1, 4)); ax[1].set_xticklabels(["Level", "Slope", "Curvature"])
ax[1].set_ylabel("% of variance"); ax[1].set_title(f"3 factors = {ex.sum():.1%}")
plt.tight_layout(); plt.savefig(FIG / "fig1_pca_loadings.png", bbox_inches="tight"); plt.close()

# 2. Stability: rolling 2y PCA --------------------------------------------
roll = {}
for d in dy.resample("ME").last().index:
    h = dy.loc[:d].iloc[-504:]
    if len(h) == 504:
        roll[d] = fit_pca(h)[1]
roll = pd.DataFrame(roll).T
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.stackplot(roll.index, (roll * 100).T.values, labels=roll.columns, colors=C[:3], alpha=0.85)
ax.set_ylim(70, 100); ax.set_ylabel("% variance (rolling 2y)")
ax.set_title("Factor structure is stable in total, but the level/slope split shifts with policy regimes")
ax.legend(loc="lower left", ncol=3)
plt.tight_layout(); plt.savefig(FIG / "fig2_rolling_pca.png", bbox_inches="tight"); plt.close()

# 3. Hedge test: does the PCA fly really strip out level and slope? -------
def daily_fly_change(weights, wings, belly):
    """Change in fly from t to t+1 using the weights known at t (out of sample)."""
    d = y.diff().shift(-1) * 100
    return d[belly] - weights["a"] * d[wings[0]] - weights["b"] * d[wings[1]]

level = (y[PCA_TENORS].diff().shift(-1) * 100).mean(axis=1)
slope = (y[10.0] - y[2.0]).diff().shift(-1) * 100
hedge_rows = []
for name, wt in [("PCA-weighted (walk-forward)", "walk_forward"), ("50/50 DV01", "fifty_fifty")]:
    W = monthly_weights(y, Params(weighting=wt))
    df = pd.concat([daily_fly_change(W, (2.0, 10.0), 5.0), level, slope], axis=1,
                   keys=["fly", "level", "slope"]).dropna()
    df = df[df.index >= "2008-03-01"]  # same out-of-sample window for both
    X = np.column_stack([np.ones(len(df)), df.level, df.slope])
    beta, *_ = np.linalg.lstsq(X, df.fly, rcond=None)
    resid = df.fly - X @ beta
    hedge_rows.append({"fly": name, "beta_level": beta[1], "beta_slope": beta[2],
                       "R2_level_slope": 1 - resid.var() / df.fly.var(),
                       "ann_vol_bp": df.fly.std() * np.sqrt(252)})
hedge = pd.DataFrame(hedge_rows).set_index("fly").round(3)
hedge.to_csv(RES / "hedge_test.csv")
out["hedge_test"] = hedge.to_dict(orient="index")

# 4. Predictability: does today's z-score predict the fly's future move? ---
W = monthly_weights(y, Params())
pred = {}
for h in [1, 5, 20, 60]:
    zs, fut = [], []
    a, b = W["a"].values, W["b"].values
    Y = y[[5.0, 2.0, 10.0]].values
    for i in range(126, len(y) - h, 5):        # every 5th day to limit overlap
        if np.isnan(a[i]):
            continue
        f = 100 * (Y[i - 125:i + 1, 0] - a[i] * Y[i - 125:i + 1, 1] - b[i] * Y[i - 125:i + 1, 2])
        zs.append((f[-1] - f.mean()) / f.std(ddof=1))
        f_fut = 100 * (Y[i + h, 0] - a[i] * Y[i + h, 1] - b[i] * Y[i + h, 2])
        fut.append(f_fut - f[-1])
    pred[f"{h}d"] = round(float(np.corrcoef(zs, fut)[0, 1]), 3)
out["corr_z_vs_future_fly_change"] = pred

# 5. Backtests -------------------------------------------------------------
base = Params()
d_wf, t_wf = run(y, base)
d_is, t_is = run(y, Params(weighting="in_sample"))
d_ff, t_ff = run(y, Params(weighting="fifty_fifty"))
# Compare all three on the SAME out-of-sample window (walk-forward needs 2y to start)
t0 = d_wf.index[0]
d_is, t_is = d_is.loc[t0:], t_is[t_is.entry >= t0]
d_ff, t_ff = d_ff.loc[t0:], t_ff[t_ff.entry >= t0]
summ = pd.DataFrame({
    "PCA walk-forward (base)": summarise(d_wf, t_wf),
    "PCA in-sample (look-ahead!)": summarise(d_is, t_is),
    "50/50 DV01 fly": summarise(d_ff, t_ff),
}).T
summ.to_csv(RES / "backtest_summary.csv")
t_wf.to_csv(RES / "trades_base.csv", index=False)

gross = d_wf.pnl_price.sum() + d_wf.pnl_carry.sum()
leg_units = (2 * (1 + t_wf.a.abs() + t_wf.b.abs())).sum()
out["breakeven_half_spread_bp"] = round(gross / leg_units, 3)
out["cost_sensitivity_sharpe"] = {
    c: summarise(*run(y, Params(half_spread_bp=c)))["sharpe"] for c in [0.0, 0.1, 0.25, 0.5]}

yearly = d_wf.groupby(d_wf.index.year)[["pnl_price", "pnl_carry", "cost", "pnl"]].sum().round(1)
yearly.to_csv(RES / "pnl_by_year.csv")

stress = []
for k, (s, e) in STRESS.items():
    w = d_wf.loc[s:e]
    stress.append({"window": k, "net_pnl_bp": round(w.pnl.sum(), 1),
                   "days_in_trade": int((w.pos != 0).sum())})
stress = pd.DataFrame(stress).set_index("window"); stress.to_csv(RES / "stress_windows.csv")

# half-life: whole sample vs within each year (the regime point)
f_is = fly_series(y, *[float(v) for v in t_is[["a", "b"]].iloc[0]])
out["half_life_full_sample_days"] = round(half_life(f_is.values), 0)
out["half_life_median_within_year_days"] = round(float(np.median(
    [half_life(g.values) for _, g in f_is.groupby(f_is.index.year) if len(g) > 50])), 0)

# 6. Robustness grid -------------------------------------------------------
grid = []
for wings, belly, lab in [((2.0, 10.0), 5.0, "2s5s10s"), ((5.0, 30.0), 10.0, "5s10s30s")]:
    for zw in [63, 126, 252]:
        for e in [1.0, 1.5, 2.0]:
            for c in [0.0, 0.25]:
                r = summarise(*run(y, Params(wings=wings, belly=belly, z_window=zw, entry=e, half_spread_bp=c)))
                grid.append({"fly": lab, "z_window": zw, "entry": e, "half_spread_bp": c,
                             "sharpe": r["sharpe"], "ann_pnl_bp": r["ann_pnl_bp"], "trades": r["trades"]})
grid = pd.DataFrame(grid); grid.to_csv(RES / "robustness_grid.csv", index=False)
out["grid_share_positive_net"] = round(float((grid[grid.half_spread_bp == 0.25].sharpe > 0).mean()), 2)
out["grid_median_sharpe_gross"] = float(grid[grid.half_spread_bp == 0].sharpe.median())
out["grid_median_sharpe_net"] = float(grid[grid.half_spread_bp == 0.25].sharpe.median())

fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, (lab, c) in zip(axs, [(f, c) for f in ["2s5s10s", "5s10s30s"] for c in [0.0, 0.25]]):
    m = grid[(grid.fly == lab) & (grid.half_spread_bp == c)].pivot(index="z_window", columns="entry", values="sharpe")
    im = ax.imshow(m.values, cmap="RdBu", vmin=-0.6, vmax=0.6)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(m.columns); ax.set_yticks(range(3)); ax.set_yticklabels(m.index)
    ax.set_xlabel("entry |z|"); ax.set_ylabel("z window (days)")
    ax.set_title(f"{lab}, {'gross' if c == 0 else 'net 0.25bp'}", fontsize=10); ax.grid(False)
fig.suptitle("Sharpe ratio across every parameter set tested (nothing hidden)", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "fig5_robustness.png", bbox_inches="tight"); plt.close()

# 3b / 4. Charts: fly + z, equity curves -----------------------------------
fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
f_wf = fly_series(y, W["a"], W["b"]).dropna()
ax[0].plot(f_wf.index, f_wf, color=C[0], lw=0.8)
ax[0].set_ylabel("bp"); ax[0].set_title("2s5s10s PCA fly (walk-forward weights) and trading signal")
ax[1].plot(d_wf.index, d_wf.z, color="grey", lw=0.5)
for s_, col in [(1, C[2]), (-1, C[3])]:
    m = d_wf.pos == s_
    ax[1].scatter(d_wf.index[m], d_wf.z[m], s=1.5, color=col, label="long fly" if s_ == 1 else "short fly")
for lvl in [base.entry, -base.entry]:
    ax[1].axhline(lvl, color="k", ls="--", lw=0.7)
ax[1].set_ylabel("z-score (6m)"); ax[1].legend(loc="lower left", markerscale=6)
plt.tight_layout(); plt.savefig(FIG / "fig3_fly_and_signal.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(d_wf.index, (d_wf.pnl_price + d_wf.pnl_carry).cumsum(), color=C[0], ls="--", label="Walk-forward, gross")
ax.plot(d_wf.index, d_wf.pnl.cumsum(), color=C[0], label="Walk-forward, net of costs (base case)")
ax.plot(d_is.index, d_is.pnl.cumsum(), color=C[1], label="In-sample weights, net (look-ahead)")
ax.plot(d_wf.index, d_wf.pnl_carry.cumsum(), color=C[2], label="Carry + roll-down only")
for k, (s, e) in STRESS.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("cumulative P&L (bp x belly DV01)")
ax.set_title("Cumulative P&L, 2s5s10s mean-reversion (grey = stress windows)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig4_pnl.png", bbox_inches="tight"); plt.close()

# Save + print -------------------------------------------------------------
with open(RES / "key_numbers.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
pd.set_option("display.width", 200)
print(json.dumps(out, indent=2, default=str))
print(summ.to_string()); print(yearly.to_string()); print(stress.to_string())
