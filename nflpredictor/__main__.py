"""Command line entry point.

  python -m nflpredictor fetch      [--seasons 2023 2024 2025 2026] [--refresh]
  python -m nflpredictor backtest   [--seasons 2023 2024 2025 2026] [--features rank_diff ...]
  python -m nflpredictor forecast   --season 2026 --week 9 [--features rank_diff ...]
  python -m nflpredictor compare    --season 2026 --week 9 [--points-per-5pct 1.0]
  python -m nflpredictor llm-forecast --season 2026 --week 9 [--mode blind informed] [--temperature 1.0]

Features are registered in features.py. nw_rank_diff uses data/nick_wright_tiers.csv.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", module="urllib3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data  # noqa: E402
from .agent import PredictorAgent, run  # noqa: E402
from .evaluate import (POINTS_PER_5PCT, ats_picks, ats_score, bootstrap,  # noqa: E402
                       calibration, expected_calibration_error, score)
from .features import FEATURES as REGISTERED  # noqa: E402

RESULTS = data.ROOT / "results"
SEASONS = [2023, 2024, 2025, 2026]
FEATURES = ["rank_diff"]


def cmd_fetch(a):
    df, log = data.build_rankings(a.seasons, refresh=a.refresh)
    if a.refresh:
        data.load_games(refresh=True)
    print("\n".join(log))
    print(f"\n{len(df)} team-week rankings -> {data.RANKINGS_CSV.relative_to(data.ROOT)}")


def cmd_backtest(a):
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    agent = PredictorAgent(a.features)
    trace = RESULTS / "trace_backtest.jsonl"
    dec = run(agent, data.load_games(), data.load_rankings(), a.seasons, trace_path=trace,
              nw_tiers=data.load_nick_tiers() if "nw_rank_diff" in a.features else None)
    dec = ats_picks(dec, a.points_per_5pct)
    dec.to_csv(RESULTS / "predictions_backtest.csv", index=False)
    s, cal, boot, ats = score(dec), calibration(dec), bootstrap(dec), ats_score(dec)
    elapsed = time.time() - t0

    lines = [
        "# Backtest summary", "",
        f"Seasons: {', '.join(map(str, a.seasons))}. Features: {', '.join(a.features)}. "
        f"Walk-forward: each week is predicted using only games completed before it.", "",
        "## Scores (completed, non-tied games)", "",
        s.to_markdown(index=False, floatfmt=".3f"), "",
        "Note: the higher-rank baseline is scored on accuracy; its fixed 0.6/0.4 "
        "probability only makes Brier/log loss defined.", "",
        "## Agent calibration (all seasons)", "",
        cal.to_markdown(index=False, floatfmt=".3f"), "",
        f"Expected calibration error: {expected_calibration_error(cal):.3f}", "",
        "## Bootstrap 95% intervals (all seasons)", "",
        "2,000 resamples of games with replacement, seed 0, percentile intervals. "
        "For Brier differences, negative means the agent is better.", "",
        boot.to_markdown(index=False, floatfmt=".3f"), "",
        "## Against the spread", "",
        f"Rule: {a.points_per_5pct:g} point(s) of predicted margin per 5 percentage points of "
        "confidence above 50%; pick the side the predicted margin favors relative to the closing "
        "spread. Pushes and games without a line are excluded. Break-even at -110 odds is 52.4%.", "",
        ats.to_markdown(index=False, floatfmt=".3f"), "",
        "## Final posterior (logit scale)", "",
        *[f"- {k}: mean {v['mean']:+.3f}, sd {v['sd']:.3f}" for k, v in agent.posterior().items()], "",
        f"Runtime: {elapsed:.2f} s for {len(dec)} decisions. Trace: `{trace.relative_to(data.ROOT)}`.",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def week_decisions(season: int, week: int, features: list[str]) -> pd.DataFrame:
    """The local agent's decisions for one week, walked forward through every earlier week."""
    tiers = data.load_nick_tiers()
    use_nw = "nw_rank_diff" in features
    if use_nw and tiers[(tiers.season == season) & (tiers.week == week)].empty:
        raise SystemExit(f"No Nick Wright tiers for {season} week {week} in "
                         f"{data.NW_TIERS_CSV.relative_to(data.ROOT)}. Enter them first.")
    dec = run(PredictorAgent(features), data.load_games(), data.load_rankings(),
              [s for s in SEASONS if s <= season], nw_tiers=tiers if use_nw else None)
    wk = dec[(dec.season == season) & (dec.week == week)]
    if wk.empty:
        raise SystemExit(f"No ranked games for {season} week {week}. Run `fetch` first?")
    return wk


def spread_label(games: pd.DataFrame) -> pd.Series:
    """'KC -9.5' style line; nflverse spread_line > 0 means the home team is favored."""
    fav = games.home_team.where(games.spread_line >= 0, games.away_team)
    return fav + " -" + games.spread_line.abs().map("{:g}".format)


def cmd_forecast(a):
    RESULTS.mkdir(exist_ok=True)
    use_nw = "nw_rank_diff" in a.features
    wk = ats_picks(week_decisions(a.season, a.week, a.features), a.points_per_5pct)
    wk = wk.assign(spread=spread_label(wk))
    nw = ["away_nw_rank", "home_nw_rank"] if use_nw else []
    cols = ["away_team", "home_team", "away_rank", "home_rank", *nw, "p_home", "pick", "confidence",
            "pred_margin", "spread", "ats_pick", "ats_edge"]
    table = wk[cols].sort_values("confidence", ascending=False).astype({"away_rank": int, "home_rank": int})
    table[nw] = table[nw].apply(lambda c: c.map("{:g}".format))  # eliminated tier can be x.5
    tag = "" if a.features == FEATURES else "_" + "+".join(a.features)  # keep each model's file
    out = RESULTS / f"forecast_{a.season}_w{a.week:02d}{tag}.md"
    text = (f"# Forecast: {a.season} week {a.week}\n\n"
            f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}, before kickoff. "
            f"Features: {', '.join(a.features)}. Inputs: NFL.com Week {a.week} Power Rankings"
            f"{', Nick Wright tiers' if use_nw else ''} and the agent's posterior after "
            f"all completed games.\n\n"
            f"ATS rule: {a.points_per_5pct:g} point(s) of predicted margin (`pred_margin`, home team) "
            f"per 5 percentage points of confidence above 50%, compared with the spread in "
            f"`data/games.csv`; `ats_edge` is the gap in points.\n\n"
            f"{table.to_markdown(index=False, floatfmt='.3f')}\n")
    out.write_text(text)
    print(text)


COMPARE_MODELS = {"Power rankings": ["rank_diff"], "Nick's tiers": ["nw_rank_diff"],
                  "Combined": ["rank_diff", "nw_rank_diff"]}


def cmd_compare(a):
    """One table: every model's winner pick and ATS pick for one week, in kickoff order."""
    import json

    RESULTS.mkdir(exist_ok=True)
    games = data.load_games()
    games = games[(games.season == a.season) & (games.week == a.week)].sort_values(["gameday", "gametime"])
    probs, notes = {}, []
    for name, feats in COMPARE_MODELS.items():
        wk = week_decisions(a.season, a.week, feats)
        probs[name] = wk.set_index(["away_team", "home_team"]).p_home
        notes.append(f"- {name}: local Bayesian agent, features `{'+'.join(feats)}`.")
    for mode in ("blind", "informed"):  # latest saved run of each LLM mode, if any
        traces = sorted(RESULTS.glob(f"trace_{a.season}_w{a.week:02d}_llm-{mode}_t*.jsonl"),
                        key=lambda f: f.stat().st_mtime)
        if not traces:
            notes.append(f"- LLM {mode}: no saved run; run `llm-forecast` first.")
            continue
        steps = [json.loads(line) for line in traces[-1].read_text().splitlines()]
        probs[f"LLM {mode}"] = pd.Series({tuple(t["game"].split("@")): t["p_home"] for t in steps})
        notes.append(f"- LLM {mode}: `{traces[-1].name}` ({steps[0]['model']}, "
                     f"temperature {steps[0]['temperature']:g}).")
    table = pd.DataFrame({"Game": games.away_team + " @ " + games.home_team
                          + np.where(games.location == "Neutral", " (neutral)", ""),
                          "Line": spread_label(games)})
    for name, p_home in probs.items():
        d = ats_picks(games.assign(p_home=[p_home.get((r.away_team, r.home_team), np.nan)
                                           for r in games.itertuples()]), a.points_per_5pct)
        conf = np.maximum(d.p_home, 1 - d.p_home)
        win = np.where(d.p_home >= 0.5, d.home_team, d.away_team)
        table[f"{name}"] = np.where(d.p_home.isna(), "", [f"{t} {c:.2f}" for t, c in zip(win, conf)])
        table[f"{name} ATS"] = d.ats_pick.fillna("")
    out = RESULTS / f"forecast_{a.season}_w{a.week:02d}_comparison.md"
    text = (f"# Forecast comparison: {a.season} week {a.week}\n\n"
            f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Games in kickoff order. Each model has a "
            f"winner pick with its win probability, then its ATS pick: {a.points_per_5pct:g} point(s) "
            f"of predicted margin per 5 percentage points of confidence above 50%, against the line "
            f"in `data/games.csv`. A blank ATS pick means the predicted margin equals the line.\n\n"
            + "\n".join(notes) + "\n\n" + table.to_markdown(index=False) + "\n")
    out.write_text(text)
    print(text)


def cmd_llm_forecast(a):
    import json

    from .llm import LLMForecaster, OpenAICompatibleModel, week_inputs

    RESULTS.mkdir(exist_ok=True)
    env = week_inputs(data.load_games(), data.load_rankings(), data.load_nick_tiers(), a.season, a.week)
    if env.empty:
        raise SystemExit(f"No games for {a.season} week {a.week}.")
    if "informed" in a.mode and env.home_rank.isna().all():
        raise SystemExit(f"No NFL.com rankings for {a.season} week {a.week}. Run `fetch` first?")
    if "informed" in a.mode and env.home_nw_rank.isna().all():
        print(f"note: no Nick Wright tiers for {a.season} week {a.week}; informed prompts omit them")
    model = OpenAICompatibleModel(a.model)
    for mode in a.mode:
        dec, trace = LLMForecaster(model, mode, a.temperature).decide(env)
        stem = f"{a.season}_w{a.week:02d}_llm-{mode}_t{a.temperature:g}"
        with open(RESULTS / f"trace_{stem}.jsonl", "w") as f:
            for step in trace:
                f.write(json.dumps(step) + "\n")
        cols = ["away_team", "home_team", "p_home", "pick", "confidence", "reason"]
        table = dec[cols].sort_values("confidence", ascending=False)
        told = ("only the matchup" if mode == "blind" else
                "the matchup, NFL.com power rank and record, and Nick Wright's tier")
        text = (f"# LLM forecast ({mode}): {a.season} week {a.week}\n\n"
                f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}, before kickoff. Model: {model.name}, "
                f"temperature {a.temperature:g}. The model was told {told}.\n\n"
                f"{table.to_markdown(index=False, floatfmt='.3f')}\n")
        abstained = int(dec.p_home.isna().sum())
        if abstained:
            text += f"\nAbstained on {abstained} game(s): no usable reply. See `trace_{stem}.jsonl`.\n"
        (RESULTS / f"forecast_{stem}.md").write_text(text)
        print(text)


def main():
    p = argparse.ArgumentParser(prog="nflpredictor")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    f.add_argument("--refresh", action="store_true", help="re-download pages and game results")
    b = sub.add_parser("backtest")
    b.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    c = sub.add_parser("forecast")
    c.add_argument("--season", type=int, required=True)
    c.add_argument("--week", type=int, required=True)
    for sp in (b, c):
        sp.add_argument("--features", nargs="+", default=FEATURES, choices=sorted(REGISTERED))
    cmp = sub.add_parser("compare", help="one table of every model's winner and ATS picks for a week")
    cmp.add_argument("--season", type=int, required=True)
    cmp.add_argument("--week", type=int, required=True)
    for sp in (b, c, cmp):
        sp.add_argument("--points-per-5pct", type=float, default=POINTS_PER_5PCT,
                        help="ATS rule: predicted points per 5 percentage points of confidence")
    lf = sub.add_parser("llm-forecast", help="ask an LLM to forecast one week (live weeks only)")
    lf.add_argument("--season", type=int, required=True)
    lf.add_argument("--week", type=int, required=True)
    lf.add_argument("--mode", nargs="+", default=["blind", "informed"], choices=["blind", "informed"])
    lf.add_argument("--temperature", type=float, default=1.0,
                    help="0 = near-deterministic; higher = more volatile (max 2)")
    lf.add_argument("--model", help="override LLM_MODEL")
    a = p.parse_args()
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "forecast": cmd_forecast,
     "compare": cmd_compare, "llm-forecast": cmd_llm_forecast}[a.cmd](a)


if __name__ == "__main__":
    main()
