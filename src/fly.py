"""
Butterfly construction, carry and roll-down.

A butterfly ("fly") is three legs: the belly against two wings, e.g. 2s5s10s:
    fly = y5 - a*y2 - b*y10          (all weights in DV01 / risk terms)

Long the fly (s = +1) = long belly DV01 1, short a DV01 of 2y and b DV01 of 10y.
It makes money when the fly yield FALLS (belly richens vs wings).

The whole question is how to choose a and b:
  * 50/50 fly: a = b = 0.5. DV01-neutral, so it is hedged against a parallel move,
    but NOT against slope moves, which dominate in hiking/cutting cycles.
  * PCA fly: choose a, b so the fly has zero loading on PC1 AND PC2. What is left
    is (mostly) curvature + bond-specific noise, which is what we want to trade.
"""
import numpy as np
import pandas as pd


def pca_fly_weights(loadings: pd.DataFrame, wings=(2.0, 10.0), belly=5.0):
    """Solve for wing weights (a, b) that neutralise PC1 and PC2.

    For each PC k:  L_belly,k = a * L_wing1,k + b * L_wing2,k
    Two equations, two unknowns.
    """
    L = loadings.iloc[:, :2]
    A = np.array([[L.loc[wings[0], c], L.loc[wings[1], c]] for c in L.columns])
    rhs = np.array([L.loc[belly, c] for c in L.columns])
    a, b = np.linalg.solve(A, rhs)
    return float(a), float(b)


def fly_series(y: pd.DataFrame, a: float, b: float, wings=(2.0, 10.0), belly=5.0):
    """Fly yield in bp, given yields in percent."""
    return 100 * (y[belly] - a * y[wings[0]] - b * y[wings[1]])


# ----------------------------------------------------------------------------
# Carry and roll-down
# ----------------------------------------------------------------------------
# Holding a bond for a day earns two things on top of price moves:
#   carry     : coupon/yield income minus the cost of financing it (repo).
#               We proxy repo with the 3-month bill yield.
#   roll-down : the bond ages, so it is priced off a shorter point on the curve.
#               On an upward-sloping curve its yield falls -> price gain.
# Both are converted into "bp of yield" per unit of DV01, so they add directly
# to the yield-change P&L.

def par_mod_duration(y_pct, T):
    """Modified duration of a par bond with semi-annual coupons (years)."""
    y = np.asarray(y_pct, dtype=float) / 100.0
    y = np.where(np.abs(y) < 1e-8, 1e-8, y)
    return (1 - (1 + y / 2) ** (-2 * T)) / y


def carry_roll_bp_per_day(y: pd.DataFrame, T: float, repo_col: float = 0.25,
                          roll_step: float = 0.25) -> pd.Series:
    """Daily carry + roll-down, in bp of yield, for a LONG position of DV01 = 1.

    carry (bp/yr) = (y_T - repo) [in bp] / duration_T
        (yield income per unit price, divided by duration to express it as an
         equivalent yield move)
    roll  (bp/yr) = [y(T) - y(T - roll_step)] / roll_step, in bp
        (local slope of the curve; linear interpolation between CMT points)
    """
    tenors = np.array(y.columns, dtype=float)
    yT = y[T]
    dur = par_mod_duration(yT, T)
    carry = 100 * (yT - y[repo_col]) / dur
    y_short = y.apply(lambda row: np.interp(T - roll_step, tenors, row.values), axis=1)
    roll = 100 * (yT - y_short) / roll_step
    return (carry + roll) / 252.0
