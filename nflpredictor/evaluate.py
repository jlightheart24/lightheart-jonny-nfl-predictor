"""Scoring the agent's decisions against baselines."""
from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd

HOME_WIN_RATE = 0.55  # long-run NFL home win rate, used by the home-team baseline
SPREAD_SD = 13.5      # std. dev. of NFL score margin around the closing spread


def _norm_cdf(x):
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x) / sqrt(2)))


def _scores(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def score(decisions: pd.DataFrame) -> pd.DataFrame:
    """Per-season and overall scores for the agent and each baseline (completed, non-tied games)."""
    d = decisions[decisions.result.notna() & (decisions.result != 0)].copy()
    d["y"] = (d.result > 0).astype(float)
    # Higher-ranked team wins; a 0.5 + tiny margin probability keeps Brier/log loss defined.
    d["p_rank"] = np.where(d.home_rank < d.away_rank, 0.6, 0.4)
    d["p_home_only"] = HOME_WIN_RATE
    # nflverse spread_line > 0 means the home team is favored by that many points.
    d["p_vegas"] = _norm_cdf(d.spread_line / SPREAD_SD)

    rows = []
    for label, grp in [*((str(s), g) for s, g in d.groupby("season")), ("all", d)]:
        y = grp.y.to_numpy()
        for name, col in [("agent (rankings, Bayesian)", "p_home"),
                          ("baseline: higher NFL.com rank", "p_rank"),
                          ("baseline: home team", "p_home_only"),
                          ("reference: Vegas closing spread", "p_vegas")]:
            rows.append({"season": label, "method": name, "games": len(grp),
                         **_scores(grp[col].to_numpy(), y)})
    return pd.DataFrame(rows)


POINTS_PER_5PCT = 1.0  # ATS rule: predicted margin per 5 percentage points of confidence above 50%


def ats_picks(decisions: pd.DataFrame, points_per_5pct: float = POINTS_PER_5PCT) -> pd.DataFrame:
    """Pick against the spread from the win model.

    Confidence above 50% becomes a predicted margin, points_per_5pct points per 5 percentage
    points (60% -> 2 points at the default). The pick is home if that predicted home margin beats
    the spread, away if it falls short, and none if they are equal.
    """
    d = decisions.copy()
    d["pred_margin"] = ((d.p_home - 0.5) * 100 / 5 * points_per_5pct).round(6)  # 0.6 -> 2.0, not 1.999…
    edge = d.pred_margin - d.spread_line
    d["ats_pick"] = np.where(edge > 0, d.home_team, np.where(edge < 0, d.away_team, None))
    d["ats_edge"] = edge.abs()
    return d


def ats_score(decisions: pd.DataFrame, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Cover rate of the ATS picks and of the always-favorite / always-underdog baselines.

    Scored on completed games with a line, excluding pushes and games with no ATS pick.
    Break-even at standard -110 odds is 52.4%. The interval is a bootstrap 95% percentile interval.
    """
    d = decisions[decisions.result.notna() & decisions.spread_line.notna()
                  & decisions.ats_pick.notna()]
    d = d[d.result != d.spread_line]
    home_covers = (d.result > d.spread_line).to_numpy()
    picks = {"agent ATS pick": (d.ats_pick == d.home_team).to_numpy(),
             "baseline: favorite": (d.spread_line > 0).to_numpy(),
             "baseline: underdog": (d.spread_line < 0).to_numpy()}
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    rows = []
    for label, m in [*((str(s), (d.season == s).to_numpy()) for s in sorted(d.season.unique())),
                     ("all", np.ones(len(d), bool))]:
        for name, pick_home in picks.items():
            won = (pick_home == home_covers).astype(float)
            row = {"season": label, "method": name, "games": int(m.sum()),
                   "cover_rate": float(won[m].mean())}
            if label == "all":
                row["ci_low"], row["ci_high"] = np.percentile(won[idx].mean(axis=1), [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def calibration(decisions: pd.DataFrame, bins=(0.5, 0.6, 0.7, 0.8, 1.0)) -> pd.DataFrame:
    """Observed win rate of the agent's pick, by stated confidence."""
    d = decisions[decisions.result.notna() & (decisions.result != 0)].copy()
    d["correct"] = ((d.p_home >= 0.5) == (d.result > 0)).astype(float)
    d["bin"] = pd.cut(d.confidence, bins=list(bins), include_lowest=True)
    return d.groupby("bin", observed=True).agg(games=("correct", "size"),
                                               mean_confidence=("confidence", "mean"),
                                               win_rate=("correct", "mean")).reset_index()


def expected_calibration_error(cal: pd.DataFrame) -> float:
    """Game-weighted mean gap between stated confidence and observed win rate, over the bins."""
    w = cal.games / cal.games.sum()
    return float(np.sum(w * np.abs(cal.mean_confidence - cal.win_rate)))


def bootstrap(decisions: pd.DataFrame, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """95% percentile intervals from resampling games with replacement (fixed seed, repeatable).

    Rows: the agent's accuracy and Brier score, and the agent's Brier minus each baseline's on the
    same resampled games (negative means the agent is better).
    """
    d = decisions[decisions.result.notna() & (decisions.result != 0)]
    y = (d.result > 0).to_numpy(float)
    p = d.p_home.to_numpy()
    others = {"higher NFL.com rank": np.where(d.home_rank < d.away_rank, 0.6, 0.4),
              "home team": np.full(len(d), HOME_WIN_RATE),
              "Vegas closing spread": _norm_cdf(d.spread_line.to_numpy() / SPREAD_SD)}
    err = (p - y) ** 2
    stats = {"agent accuracy": ((p >= 0.5) == (y == 1)).astype(float), "agent Brier": err,
             **{f"agent Brier − {k}": err - (q - y) ** 2 for k, q in others.items()}}
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    rows = []
    for name, v in stats.items():
        means = v[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        rows.append({"statistic": name, "estimate": float(v.mean()), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
