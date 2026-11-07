"""The prediction agent and its week-by-week environment loop.

Each NFL week is one step:
  1. PERCEIVE  the power rankings published before the week, plus that week's schedule
               (teams and home/away only; scores are hidden until the week is played).
  2. DECIDE    a home-win probability for every game from the current posterior.
  3. ACT       emit a pick and probability per game (logged to the trace).
  4. OBSERVE   final scores once the week is played.
  5. UPDATE    the posterior over model weights using every game observed so far.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import PRIORS, design_matrix
from .model import BayesLogit


@dataclass
class PredictorAgent:
    features: list[str]
    model: BayesLogit = None
    seen: list = field(default_factory=list)  # (X rows, y) observed so far

    def __post_init__(self):
        names = ["home_field"] + self.features
        self.model = BayesLogit([PRIORS[n][0] for n in names], [PRIORS[n][1] for n in names])

    def decide(self, games: pd.DataFrame) -> pd.DataFrame:
        X = design_matrix(games, self.features)
        p = self.model.predict(X)
        cols = ["game_id", "season", "week", "away_team", "home_team", "away_rank", "home_rank"]
        cols += [c for c in ("away_nw_rank", "home_nw_rank") if c in games]
        out = games[cols].copy()
        out["p_home"] = p
        out["pick"] = np.where(p >= 0.5, out.home_team, out.away_team)
        out["confidence"] = np.maximum(p, 1 - p)
        return out

    def observe(self, games: pd.DataFrame):
        done = games[games.result.notna() & (games.result != 0)]  # ties carry no win label
        if len(done):
            self.seen.append((design_matrix(done, self.features), (done.result > 0).to_numpy(float)))
            X = np.vstack([s[0] for s in self.seen])
            y = np.concatenate([s[1] for s in self.seen])
            self.model.fit(X, y)

    def posterior(self) -> dict:
        names = ["home_field"] + self.features
        return {n: {"mean": round(float(m), 4), "sd": round(float(s), 4)}
                for n, m, s in zip(names, self.model.mean, self.model.sd())}


def attach_rankings(games: pd.DataFrame, rankings: pd.DataFrame, col: str = "rank") -> pd.DataFrame:
    """Join the rankings published *entering* each week onto that week's games.

    Adds home_{col} and away_{col}, e.g. home_rank or (with col="nw_rank") home_nw_rank.
    """
    r = rankings[["season", "week", "team", "rank"]]
    g = games.merge(r.rename(columns={"team": "home_team", "rank": f"home_{col}"}),
                    on=["season", "week", "home_team"], how="left")
    g = g.merge(r.rename(columns={"team": "away_team", "rank": f"away_{col}"}),
                on=["season", "week", "away_team"], how="left")
    return g


def run(agent: PredictorAgent, games: pd.DataFrame, rankings: pd.DataFrame,
        seasons: list[int], trace_path=None, nw_tiers: pd.DataFrame = None) -> pd.DataFrame:
    """Walk the agent forward through every week of the given seasons.

    nw_tiers (optional) are Nick Wright tier ranks from data.load_nick_tiers().
    """
    env = attach_rankings(games[games.season.isin(seasons)], rankings)
    if nw_tiers is not None:
        env = attach_rankings(env, nw_tiers, col="nw_rank")
    decisions, trace = [], []
    for (season, week), wk in env.groupby(["season", "week"], sort=True):
        visible = wk.dropna(subset=["home_rank", "away_rank"])
        step = {"season": int(season), "week": int(week), "games": len(wk),
                "ranked_games": len(visible), "prior": agent.posterior()}
        if len(visible) == 0:
            step["action"] = "abstain: no power rankings published for this week"
            trace.append(step)
            continue
        d = agent.decide(visible)
        d = d.merge(wk[["game_id", "result", "spread_line", "away_score", "home_score"]], on="game_id")
        decisions.append(d)
        step["decisions"] = [
            {"game": f"{r.away_team}@{r.home_team}", "ranks": f"{int(r.away_rank)}@{int(r.home_rank)}",
             **({"nw_ranks": f"{r.away_nw_rank:g}@{r.home_nw_rank:g}"}
                if "home_nw_rank" in d and pd.notna(r.home_nw_rank) else {}),
             "p_home": round(float(r.p_home), 3), "pick": r.pick,
             "final": None if pd.isna(r.result) else f"{int(r.away_score)}-{int(r.home_score)}"}
            for r in d.itertuples()]
        agent.observe(visible)
        step["posterior"] = agent.posterior()
        trace.append(step)
    if trace_path:
        with open(trace_path, "w") as f:
            for step in trace:
                f.write(json.dumps(step) + "\n")
    return pd.concat(decisions, ignore_index=True)
