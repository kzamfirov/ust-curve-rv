"""
Load daily US Treasury constant-maturity (CMT) yields from FRED.

Why these series: CMT yields are the official daily par curve published by the
US Treasury and mirrored on FRED. They are free, long, and consistent, which makes
them the standard starting point for curve research. They are NOT tradeable bonds
(they are an interpolated curve), so the backtest is a research approximation of
trading on-the-run Treasuries or futures, not a P&L you could book exactly.

Sample starts 9 Feb 2006, when Treasury resumed publishing the 30-year after the
2002-2006 suspension, so every tenor is available on every day.
"""
from pathlib import Path
import urllib.request

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# FRED series id -> maturity in years
SERIES = {
    "DGS3MO": 0.25, "DGS1": 1, "DGS2": 2, "DGS3": 3, "DGS5": 5,
    "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30,
}
START = "2006-02-09"
END = "2026-09-29"   # pinned so every re-run reproduces the published numbers


def _download(series_id: str) -> Path:
    path = DATA_DIR / f"{series_id}.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        urllib.request.urlretrieve(url, path)
    return path


def load_yields(start: str = START, end: str = END) -> pd.DataFrame:
    """Return a DataFrame of yields in PERCENT, columns = maturity in years (float).

    Rows with any missing tenor (US bond-market holidays) are dropped, so every
    row is a complete curve.
    """
    frames = []
    for sid, mat in SERIES.items():
        s = pd.read_csv(_download(sid), index_col=0, parse_dates=True, na_values=".")
        s.columns = [float(mat)]
        frames.append(s)
    df = pd.concat(frames, axis=1, sort=True).loc[start:end].dropna()
    df.index.name = "date"
    return df
