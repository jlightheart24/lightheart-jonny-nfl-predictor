"""LLM forecaster: asks a chat model for each game's winner and home-win probability.

Two prompt modes, identical except for what the model is told:
  blind     only the matchup (teams, home/away, date). The model uses its own knowledge.
  informed  the matchup plus the chosen inputs: NFL.com power rank and record, and
            Nick Wright's tier (name and tie rank).

Temperature sets how volatile the agent is: 0 is close to deterministic, higher values
let repeated runs disagree more. It is recorded with every forecast.

The client talks to any OpenAI-compatible service (the course LLM service) configured by
LLM_BASE_URL, LLM_API_KEY and LLM_MODEL, from the environment or a local .env file.
The API key is never written to results or traces.

The model's training data may contain results of past seasons, so these forecasts are
only meaningful for games not yet played (live weeks), not as a backtest.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .data import ROOT, TEAMS

NAMES = {abbr: name for name, abbr in TEAMS.items()}
DEFAULT_MODEL = "Qwen3.8-Flash-Next"
MODES = ("blind", "informed")

SYSTEM = (
    "You are an NFL analyst forecasting the winner of one regular-season game. "
    "Reply with only a JSON object, no other text: "
    '{"pick": "<team abbreviation>", "p_home_win": <probability from 0 to 1 that the HOME '
    'team wins>, "reason": "<one sentence>"}'
)


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], temperature: float) -> str: ...


def _load_env(path=ROOT / ".env"):
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class OpenAICompatibleModel:
    def __init__(self, model: str | None = None):
        _load_env()
        missing = [v for v in ("LLM_BASE_URL", "LLM_API_KEY") if not os.environ.get(v)]
        if missing:
            raise SystemExit(f"Missing {', '.join(missing)}. Set them in the environment or in "
                             f".env (see .env.example).")
        from openai import OpenAI

        self.name = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self._client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                              timeout=300, max_retries=2)

    def complete(self, messages: list[dict], temperature: float) -> str:
        r = self._client.chat.completions.create(model=self.name, messages=messages,
                                                 temperature=temperature)
        return r.choices[0].message.content or ""


def week_inputs(games: pd.DataFrame, rankings: pd.DataFrame, tiers: pd.DataFrame,
                season: int, week: int) -> pd.DataFrame:
    """That week's games with the informed-mode inputs joined on (all published before kickoff)."""
    g = games[(games.season == season) & (games.week == week)]
    r = rankings[(rankings.season == season) & (rankings.week == week)]
    t = tiers[(tiers.season == season) & (tiers.week == week)]
    for side in ("home", "away"):
        g = g.merge(r[["team", "rank", "record"]].rename(columns={
            "team": f"{side}_team", "rank": f"{side}_rank", "record": f"{side}_record"}),
            on=f"{side}_team", how="left")
        g = g.merge(t[["team", "rank", "tier_name"]].rename(columns={
            "team": f"{side}_team", "rank": f"{side}_nw_rank", "tier_name": f"{side}_nw_tier"}),
            on=f"{side}_team", how="left")
    return g


def prompt(game, mode: str) -> str:
    """The user message for one game. `game` is a row of the week's environment table."""
    a, h = game.away_team, game.home_team
    day = pd.Timestamp(game.gameday).strftime("%A %Y-%m-%d")
    lines = [f"{game.season} NFL season, week {game.week}, played {day}.",
             f"Away team: {NAMES[a]} ({a}). Home team: {NAMES[h]} ({h})."]
    if mode == "informed":
        lines.append("")
        if pd.notna(game.get("home_rank")):
            rec = lambda r: f" ({r})" if isinstance(r, str) and r else ""  # noqa: E731
            lines.append(f"NFL.com week {game.week} Power Rankings (1 = best of 32): "
                         f"{a} #{int(game.away_rank)}{rec(game.away_record)}, "
                         f"{h} #{int(game.home_rank)}{rec(game.home_record)}.")
        if pd.notna(game.get("home_nw_rank")):
            lines.append(f"Nick Wright's week {game.week} NFL tiers (FS1 First Things First; teams in a "
                         f"tier share its best rank, 1 = best): "
                         f"{a} in \"{game.away_nw_tier}\" (rank {game.away_nw_rank:g}), "
                         f"{h} in \"{game.home_nw_tier}\" (rank {game.home_nw_rank:g}).")
        lines.append("")
        lines.append("Use this information along with anything else you know.")
    else:
        lines.append("Use your own knowledge of these teams.")
    lines.append(f"Which team wins? Pick {a} or {h}.")
    return "\n".join(lines)


_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse(text: str, away: str, home: str) -> dict:
    """Pull {pick, p_home_win, reason} out of a reply; raise ValueError if it is unusable."""
    body = _THINK.sub("", text or "")
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    obj = json.loads(m.group(0))
    p = float(obj["p_home_win"])
    if p > 1 and p <= 100:  # a percentage instead of a probability
        p /= 100
    if not 0 <= p <= 1:
        raise ValueError(f"p_home_win {p} is outside 0-1")
    said = str(obj.get("pick", "")).strip().upper()
    said = said if said in (away, home) else None
    # The probability is the forecast; the stated pick only breaks an exact 0.5.
    pick = home if p > 0.5 else away if p < 0.5 else (said or home)
    return {"p_home": p, "pick": pick, "reason": str(obj.get("reason", "")).strip(),
            "pick_disagreed": said is not None and said != pick}


@dataclass
class LLMForecaster:
    model: ChatModel
    mode: str = "blind"
    temperature: float = 1.0
    attempts: int = 3

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def decide(self, games: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
        """One model call per game. A game whose replies never parse is abstained (NaN)."""
        rows, trace = [], []
        for g in games.itertuples(index=False):
            g = pd.Series(g._asdict())
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(g, self.mode)}]
            out, replies = None, []
            for _ in range(self.attempts):
                reply = self.model.complete(msgs, self.temperature)
                replies.append(reply)
                try:
                    out = parse(reply, g.away_team, g.home_team)
                    break
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                    replies.append(f"[parse error: {e}]")
            if out is None:
                out = {"p_home": np.nan, "pick": None, "reason": "abstain: no usable reply",
                       "pick_disagreed": False}
            rows.append({"game_id": g.game_id, "season": g.season, "week": g.week,
                         "away_team": g.away_team, "home_team": g.home_team, **out})
            trace.append({"game": f"{g.away_team}@{g.home_team}", "mode": self.mode,
                          "model": self.model.name, "temperature": self.temperature,
                          "prompt": msgs[1]["content"], "replies": replies, **out})
        d = pd.DataFrame(rows)
        d["confidence"] = np.maximum(d.p_home, 1 - d.p_home)
        return d, trace
