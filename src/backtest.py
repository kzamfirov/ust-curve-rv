"""
Walk-forward backtest of a mean-reversion butterfly strategy.

No look-ahead rules (the single most important thing in any backtest):
  1. PCA weights for month M are estimated ONLY on data before the first day of M.
  2. The z-score on day t uses closes up to and including t.
  3. A position decided at the close of t earns the move from t to t+1.

Trading rule (deliberately simple, parameters chosen up front, not optimised):
  z = (fly_t - rolling mean) / rolling std, over the last `z_window` days
  z >= +entry : fly yield is high -> belly CHEAP vs wings -> long the fly (s = +1)
  z <= -entry : belly RICH -> short the fly (s = -1)
  exit when z mean-reverts to `exit_z`, or stop out at |z| >= `stop_z`,
  or after `max_hold` days. After a stop or time exit, the signal must fall back
  inside the entry band before a new trade (otherwise we would re-enter the same
  losing view the next morning).

P&L is in bp per 1 unit of belly DV01 (e.g. x $10,000 if the belly is sized at
$10k/bp). Weights of an open trade are frozen until it is closed, as a trader would.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fly import pca_fly_weights, carry_roll_bp_per_day
from .pca import fit_pca

PCA_TENORS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]


def half_life(x: np.ndarray) -> float:
    """Half-life (days) of mean reversion from an AR(1) fit x_t = c + phi x_{t-1} + e.

    Discrete version of an Ornstein-Uhlenbeck process: half-life = -ln 2 / ln(phi).
    phi >= 1 means no mean reversion in this window -> infinite half-life.
    """
    phi = np.polyfit(x[:-1], x[1:], 1)[0]
    return -np.log(2) / np.log(phi) if 0 < phi < 1 else np.inf


@dataclass
class Params:
    wings: tuple = (2.0, 10.0)
    belly: float = 5.0
    weighting: str = "walk_forward"  # "walk_forward" | "in_sample" | "fifty_fifty"
    pca_window: int = 504            # 2 years of daily changes for each PCA fit
    z_window: int = 126              # 6 months for the z-score
    entry: float = 1.5
    exit_z: float = 0.0
    stop_z: float = 3.0
    max_hold: int = 60
    half_spread_bp: float = 0.25     # cost per leg, per side, in bp of yield x DV01
    max_half_life: float = None      # regime filter: only enter if the fly is
                                     # currently mean-reverting fast enough (days)


def monthly_weights(y: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Wing weights (a, b) that apply on each day."""
    dy = y[PCA_TENORS].diff().mul(100).dropna()
    idx = y.index
    out = pd.DataFrame(index=idx, columns=["a", "b"], dtype=float)

    if p.weighting == "fifty_fifty":
        out[:] = 0.5
        return out
    if p.weighting == "in_sample":
        # Deliberately cheats: uses the whole sample, including the future.
        # Included only to show how much look-ahead flatters a backtest.
        L, _ = fit_pca(dy)
        out[:] = pca_fly_weights(L, p.wings, p.belly)
        return out

    month_starts = idx[~idx.to_period("M").duplicated()]
    for m in month_starts:
        hist = dy[dy.index < m].iloc[-p.pca_window:]   # strictly before month start
        if len(hist) < p.pca_window:
            continue
        L, _ = fit_pca(hist)
        out.loc[m:, ["a", "b"]] = pca_fly_weights(L, p.wings, p.belly)
    return out


def run(y: pd.DataFrame, p: Params = Params()):
    w1, w2, bl = p.wings[0], p.wings[1], p.belly
    W = monthly_weights(y, p)
    Y = y[[bl, w1, w2]].values                        # percent
    cr = np.column_stack([carry_roll_bp_per_day(y, t).values for t in (bl, w1, w2)])
    A, B = W["a"].values, W["b"].values
    n = len(y)

    pos = np.zeros(n); pnl_px = np.zeros(n); pnl_cr = np.zeros(n); cost = np.zeros(n)
    zs = np.full(n, np.nan)
    trades = []
    s, ta, tb, entry_i, armed = 0, 0.0, 0.0, None, True

    def window(i, a, b):
        sl = slice(i - p.z_window + 1, i + 1)
        return 100 * (Y[sl, 0] - a * Y[sl, 1] - b * Y[sl, 2])

    def zscore(i, a, b):
        win = window(i, a, b)
        sd = win.std(ddof=1)
        return (win[-1] - win.mean()) / sd if sd > 0 else 0.0

    first = int(np.argmax(~np.isnan(A)))
    start = max(first, p.z_window)
    for i in range(start, n - 1):
        if s == 0:
            if np.isnan(A[i]):
                continue
            z = zscore(i, A[i], B[i]); zs[i] = z
            if not armed and abs(z) < p.entry:
                armed = True
            ok_regime = (p.max_half_life is None
                         or half_life(window(i, A[i], B[i])) < p.max_half_life)
            if armed and abs(z) >= p.entry and ok_regime:
                s = 1 if z > 0 else -1
                ta, tb, entry_i = A[i], B[i], i
                cost[i] += p.half_spread_bp * (1 + abs(ta) + abs(tb))
        else:
            z = zscore(i, ta, tb); zs[i] = z
            held = i - entry_i
            reason = None
            if s * z <= p.exit_z:
                reason = "target"
            elif s * z >= p.stop_z:
                reason = "stop"
            elif held >= p.max_hold:
                reason = "time"
            if reason:
                cost[i] += p.half_spread_bp * (1 + abs(ta) + abs(tb))
                trades.append(dict(entry=y.index[entry_i], exit=y.index[i], side=s,
                                   a=ta, b=tb, days=held, reason=reason))
                armed = reason == "target"
                s = 0
        if s != 0:
            dfly = 100 * ((Y[i + 1, 0] - Y[i, 0]) - ta * (Y[i + 1, 1] - Y[i, 1])
                          - tb * (Y[i + 1, 2] - Y[i, 2]))
            pos[i] = s
            pnl_px[i] = -s * dfly
            pnl_cr[i] = s * (cr[i, 0] - ta * cr[i, 1] - tb * cr[i, 2])

    daily = pd.DataFrame(dict(pos=pos, z=zs, pnl_price=pnl_px, pnl_carry=pnl_cr,
                              cost=cost), index=y.index)
    daily["pnl"] = daily.pnl_price + daily.pnl_carry - daily.cost
    daily = daily.iloc[start:]
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["pnl"] = [daily.loc[r.entry:r.exit, "pnl"].sum() for r in tr.itertuples()]
    return daily, tr


def summarise(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    r = daily["pnl"]
    cum = r.cumsum()
    dd = (cum - cum.cummax()).min()
    vol = r.std() * np.sqrt(252)
    return {
        "start": daily.index[0].date(), "end": daily.index[-1].date(),
        "total_pnl_bp": round(r.sum(), 1),
        "ann_pnl_bp": round(r.mean() * 252, 1),
        "ann_vol_bp": round(vol, 1),
        "sharpe": round(r.mean() * 252 / vol, 2) if vol > 0 else np.nan,
        "max_drawdown_bp": round(dd, 1),
        "trades": len(trades),
        "hit_rate": round((trades["pnl"] > 0).mean(), 2) if len(trades) else np.nan,
        "avg_hold_days": round(trades["days"].mean(), 1) if len(trades) else np.nan,
        "time_in_market": round((daily["pos"] != 0).mean(), 2),
        "price_pnl_bp": round(daily["pnl_price"].sum(), 1),
        "carry_roll_pnl_bp": round(daily["pnl_carry"].sum(), 1),
        "costs_bp": round(daily["cost"].sum(), 1),
    }
