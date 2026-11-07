"""Feature registry. Each feature maps a week's games + percepts to one column.

New metrics are added here as the project grows; the agent selects them by name.
All features are oriented so a positive value favors the HOME team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_diff(games: pd.DataFrame) -> np.ndarray:
    """NFL.com power-rank advantage of the home team, per 10 ranking spots."""
    return ((games.away_rank - games.home_rank) / 10.0).to_numpy()


def nw_rank_diff(games: pd.DataFrame) -> np.ndarray:
    """Nick Wright tier-rank advantage of the home team, per 10 ranking spots.

    Weeks without hand-entered tiers contribute 0, so they carry no information about
    this weight and its posterior stays at the prior until tiers are entered.
    """
    return ((games.away_nw_rank - games.home_nw_rank) / 10.0).fillna(0.0).to_numpy()


FEATURES = {
    "rank_diff": rank_diff,
    "nw_rank_diff": nw_rank_diff,
}

# Prior (mean, sd) on the logit scale for each coefficient. Weakly informative:
# home field is centered at a small advantage; feature effects start at zero.
PRIORS = {
    "home_field": (0.15, 0.5),
    "rank_diff": (0.0, 1.0),
    "nw_rank_diff": (0.0, 1.0),
}


def design_matrix(games: pd.DataFrame, names: list[str]) -> np.ndarray:
    cols = [np.ones(len(games))] + [FEATURES[n](games) for n in names]
    return np.column_stack(cols)
