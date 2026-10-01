"""
Principal component analysis of daily yield changes.

Idea (Litterman & Scheinkman, 1991): daily moves across the whole curve are highly
correlated, so a handful of factors explain most of the variation:
  PC1 "level"     - all yields move together
  PC2 "slope"     - short end vs long end (steepening / flattening)
  PC3 "curvature" - the belly vs the wings (the "butterfly")

We run PCA on the covariance matrix of daily changes in basis points. Covariance
(not correlation) is the right choice because every column is in the same unit
(bp), and we care about how many bp each tenor actually moves.
"""
import numpy as np
import pandas as pd

PC_NAMES = ["PC1 level", "PC2 slope", "PC3 curvature"]


def fit_pca(dy_bp: pd.DataFrame, n: int = 3):
    """Eigen-decompose the covariance matrix of yield changes.

    Returns
    -------
    loadings : DataFrame (tenor x PC) - bp move in each tenor for a 1-unit move in the PC
    explained : Series - share of total variance explained by each PC
    """
    cov = np.cov(dy_bp.values, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)          # eigh: symmetric matrix, ascending order
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]

    tenors = np.array(dy_bp.columns, dtype=float)
    # Eigenvectors have arbitrary sign; fix a convention so charts are readable
    # and weights are stable through time.
    # PC1: positive on average (all yields up).
    if eigvec[:, 0].mean() < 0:
        eigvec[:, 0] *= -1
    # PC2: long end minus short end positive (a steepener is +).
    if eigvec[-1, 1] - eigvec[0, 1] < 0:
        eigvec[:, 1] *= -1
    # PC3: belly (tenor nearest 5y) negative, i.e. + = wings up vs belly.
    belly = int(np.argmin(np.abs(tenors - 5)))
    if eigvec[belly, 2] > 0:
        eigvec[:, 2] *= -1

    loadings = pd.DataFrame(eigvec[:, :n], index=tenors, columns=PC_NAMES[:n])
    explained = pd.Series(eigval[:n] / eigval.sum(), index=PC_NAMES[:n])
    return loadings, explained
