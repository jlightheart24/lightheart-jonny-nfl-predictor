# nflpredictor

A week-by-week NFL game-prediction agent for CS 4300. Before each week, the agent reads the
NFL.com Power Rankings, assigns each game a home-win probability using a Bayesian logistic
model, makes a pick, and then updates its beliefs after the games are played. The first version
uses only the media rankings. More metrics are added through `nflpredictor/features.py`.

Design: [`docs/project-definition.md`](docs/project-definition.md)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tested with Python 3.9.6 on macOS.

## Proof-of-concept run

```bash
# 1. (Optional) Re-scrape the NFL.com Power Rankings. The parsed result is already committed
#    as data/power_rankings_nflcom.csv, and the game results snapshot as data/games.csv.
.venv/bin/python -m nflpredictor fetch --seasons 2023 2024 2025 2026

# 2. Walk-forward backtest over 2023 through 2026 Week 8. Writes results/summary.md,
#    results/predictions_backtest.csv and results/trace_backtest.jsonl.
.venv/bin/python -m nflpredictor backtest

# 3. Live forecast for games that have not been played yet. Writes results/forecast_2026_w09.md.
.venv/bin/python -m nflpredictor forecast --season 2026 --week 9

# Tests
.venv/bin/python -m unittest discover tests
```

## LLM forecasts

`llm-forecast` asks a chat model on the course LLM service for each game's winner and
home-win probability, in two modes:

- `blind`: told only the matchup (teams, home/away, date). The model uses its own knowledge.
- `informed`: also told each team's NFL.com power rank and record and its Nick Wright tier.

```bash
cp .env.example .env    # then fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (git-ignored)
.venv/bin/python -m nflpredictor llm-forecast --season 2026 --week 9 --temperature 1.2
```

`--temperature` sets how volatile the agent is: 0 is close to deterministic, and higher values
(up to 2) make repeated runs disagree more. Each run writes
`results/forecast_2026_w09_llm-{mode}_t{temperature}.md` and a trace with every prompt and raw
reply. A game whose replies never parse after 3 tries is abstained. The model may have seen past
seasons' results in training, so LLM forecasts are only for weeks not yet played, not backtests.

## Comparison table and ATS picks

```bash
.venv/bin/python -m nflpredictor compare --season 2026 --week 9
```

Writes `results/forecast_2026_w09_comparison.md`: one row per game in kickoff order, with the
line and, for each model (power rankings, Nick's tiers, combined, LLM blind, LLM informed), its
winner pick with win probability and its against-the-spread pick. The LLM columns use the latest
saved `llm-forecast` run for that week. The ATS rule turns confidence above 50% into a predicted
margin, 1 point per 5 percentage points (`--points-per-5pct`), and picks the side that margin
favors relative to the spread. `forecast` and `backtest` report the same ATS pick; the backtest
scores it against the closing line.

`fetch --refresh` re-downloads the pages and the latest game results. Without it, the cached
copies are used.

## Layout

| Path | Role |
|---|---|
| `nflpredictor/data.py` | Environment data: nflverse schedule/results; NFL.com rankings scraper and validated parser |
| `nflpredictor/features.py` | Feature registry and priors (currently `rank_diff`) |
| `nflpredictor/model.py` | Bayesian logistic regression (MAP + Laplace approximation) |
| `nflpredictor/agent.py` | Agent and weekly perceive → decide → act → observe → update loop |
| `nflpredictor/llm.py` | LLM forecaster: blind and informed prompts, reply parsing, temperature |
| `nflpredictor/evaluate.py` | Accuracy, Brier score, log loss, calibration, and baselines |
| `results/` | Retained run evidence |

## Data sources

- Game schedule, scores and closing spreads: [nflverse/nfldata](https://github.com/nflverse/nfldata) `games.csv`
- Power rankings: NFL.com weekly Power Rankings articles,
  `https://www.nfl.com/news/nfl-power-rankings-week-{W}-{YEAR}-nfl-season`
- Your own rankings: `data/my_rankings.csv`, in the same columns as the NFL.com file
  (`season,week,rank,team`; one row per team; ranks 1–32, each used once). Commit each week's
  rankings before that week's first kickoff so the timestamp shows they were made in advance.
- Nick Wright's NFL tiers (First Things First): `data/nick_wright_tiers.csv`, entered by hand from
  the show (`season,week,tier,tier_name,team`; one row per team; tier 1 is the top tier). Use the
  week the tiers are released *entering*. Teams in the same tier tie for that tier's best rank:
  tier sizes 1, 2, 3 give ranks 1, 2, 2, 4, 4, 4. Teams in a tier named "Eliminated" share the
  average of their ranks instead (two teams at 31–32 both get 31.5). Used only by the opt-in `nw_rank_diff` feature,
  e.g. `forecast --season 2026 --week 10 --features rank_diff nw_rank_diff`. Weeks without tiers
  contribute nothing to that feature's weight. FOX Sports published written tiers only for 2021
  and part of 2022; since 2023 they are video only.
