# Project Definition: NFL Game-Prediction Agent

**Name:** Jonathan Lightheart
**Course:** CS 4300, Unit U0: Integrated agent systems
**Repository:** [repository URL]
**Submitted commit:** [`git rev-parse HEAD`]
**Date:** November 2026

---

## 1. Problem, user, use, and target outcome

**Problem.** Every NFL week, fans try to predict about 16 games. Most rely on gut feeling or on
media power rankings and a handful of popular stats. Rankings are free and widely read, but they
are ordinal opinions: they don't say *how likely* a team is to win a particular game, and they
ignore home field. Fans also have no easy way to tell which metrics actually predict winners
and which are noise.

**Intended user.** NFL fans who want to predict games and to understand which metrics matter
and how accurate those metrics are.

**Intended use.** Before each week's games, the agent reads that week's public power rankings
and the schedule, then publishes a pick and a win probability for every game. After the games,
it updates itself from the results, so its predictions and its estimate of each metric's value
improve over the season. The user can follow the picks, and can read the learned weight and
uncertainty of each metric to see what is driving them.

**Target outcome.**
1. *Better predictions:* probabilities that beat simple heuristics (always pick the home team;
   always pick the higher-ranked team) on Brier score, are well calibrated, and move closer to
   the betting market's closing line as more metrics are added. The market line serves as the
   reference ceiling. The numerical thresholds are in §10.
2. *Explained metrics:* for every metric, a learned weight with an uncertainty, plus the change
   in Brier score when that metric is added (§7). Together these show the user how much each
   metric is worth.

## 2. PEAS analysis

| | |
|---|---|
| **Performance measure** | Pick accuracy; Brier score and log loss of the stated probabilities; calibration (stated confidence vs. observed win rate); runtime and data-collection cost; data reliability (share of weeks parsed and validated); value of each metric (the posterior standard deviation of its weight, and its change in Brier score in ablation). Prediction measures are compared against the home-team, higher-rank and Vegas-spread baselines. |
| **Environment** | The NFL regular season, one week per step (18 weeks, about 272 games per season). Partially observable (injuries, weather and game-day events are hidden), stochastic, sequential (beliefs carry forward), and single-agent with respect to outcomes (the agent's picks do not affect games). Static during a decision, dynamic between weeks. |
| **Actuators** | Publishing, for each game: a picked winner, P(home win), and a confidence value. Written to forecast and trace files. Each week it also publishes a metrics report: every metric's learned weight, its uncertainty, and its change in Brier score. |
| **Sensors** | (1) The NFL.com weekly Power Rankings article (rank 1–32 for every team), scraped and validated. (2) The nflverse schedule (matchups, home/away). (3) Final scores, visible only after the week is played; point differential and points per game are computed from these. (4) From Nov 19: the user's own weekly rankings, `data/my_rankings.csv`. |

## 3. Agent design

- **Decision boundary.** The agent decides probabilities and picks for every ranked game in the
  upcoming week, and how much weight each metric deserves. The developer chooses which metrics
  the agent may use (registered in `features.py`). The agent learns their weights from results,
  which is why those weights can be read as evidence of which metrics matter. It does *not* place
  bets, choose its own data sources or metrics, or decide whether a human acts on its picks. If
  the rankings for a week are missing or fail validation, it abstains
  for that week rather than guessing.
- **Percepts (per week *w*).** Rankings published *entering* week *w*; the week-*w* schedule
  with home/away; after the week, final scores.
- **Actions.** For each game *g*: `pick(g)`, `p_home(g)`, `confidence(g) = max(p, 1-p)`, or
  `abstain(w)`. Once per week: `report(w)`, which publishes each metric's weight and uncertainty.
- **State.** A Gaussian posterior over model weights (mean vector and covariance), plus the set of
  games observed so far. State carries across weeks and seasons. Every completed game currently
  counts equally, so a 2023 game counts as much as last week's. This is a known weakness when the
  ranking writer changes (§9). Equal weighting is a deliberate choice for this project, and it is listed as a limitation (§10).
- **Goal / terminal conditions.** The primary goal is to minimize expected Brier score.
  Accuracy is reported alongside it but is not optimized separately. A secondary goal is to
  estimate each metric's weight precisely, which shows as a shrinking posterior standard deviation.
  One episode ends after week 18; the evaluation run ends when no published rankings
  remain (currently 2026 week 9).
- **Interaction loop.** `perceive → decide → act → observe → update`, once per week:

  1. Perceive the week-*w* rankings and schedule (scores hidden).
  2. Decide P(home win) from the posterior predictive.
  3. Act by publishing picks, probabilities and the metrics report, logged to the trace.
  4. Observe the final scores.
  5. Update the posterior using every completed game, then move to week *w+1*.

## 4. AI strategy

**Probabilistic reasoning with Bayesian logistic regression, updated week by week.**

P(home wins) = σ(β₀ + β₁·x₁ + … + βₖ·xₖ)

- β₀ is home-field advantage. x₁ = `rank_diff` = (away rank − home rank) / 10, so positive
  values favor the home team.
- Prior: β₀ ~ N(0.15, 0.5²), β₁ ~ N(0, 1²). This is weakly informative: a slight home edge and no
  assumed ranking effect. On day one, the agent doesn't yet know whether the rankings mean
  anything.
- After each week, the agent recomputes the MAP estimate with Newton's method and uses a Laplace
  approximation for the covariance. Predictions integrate over weight uncertainty with the probit
  approximation, so early-season and low-data predictions are pulled toward 50%.

**Why it fits.** Outcomes are noisy binary events, and the user needs *probabilities*, not
just picks. The Bayesian update lets the agent start from no evidence and learn how much the
rankings are worth from experience. Uncertainty shrinks as data accumulates, which the trace
shows directly. New metrics become new features with their own priors (registered in
`features.py`), so the "add metrics over time" plan fits without changing the agent. The learned
coefficients and their standard deviations also show how much each new metric adds.

**Why not the alternatives.** A neural network or gradient boosting could fit nonlinear
patterns, but with about 270 games a season they would overfit. They also give no built-in
uncertainty and can't tell the user what each metric is worth. An Elo rating on its own predicts
reasonably well, but it uses only game results, so it can't judge outside metrics. Here Elo
becomes one of the candidate metrics instead.

**Principal technical uncertainty the proof of concept had to resolve.**
(1) Can subjective, ordinal media rankings be obtained reliably, for past seasons as well as the
current week, in a form an agent can sense? (2) Do they carry enough predictive signal, once
turned into probabilities, to beat simple heuristics, and how far are they from the market?

**Uncertainty for the next phase.** All three planned metrics (rank, point differential,
points per game) measure the same thing: how good a team is. When metrics are strongly correlated, the
model can split credit between them almost arbitrarily, so a single metric's weight is not
reliable evidence that it matters. To deal with this, each metric is judged by its ablation
result (the change in Brier score when it is added, with a bootstrap confidence interval, §7)
rather than by its raw weight, and the correlations between metrics are reported alongside the
weights.

## 5. System diagram

```
 ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────────────┐
 │ NFL.com Power        │  │ Your rankings        │  │ nflverse games.csv    │
 │ Rankings (weekly     │  │ data/my_rankings.csv │  │ schedule, scores,     │
 │ HTML articles)       │  │ (from Nov 19)        │  │ spreads               │
 └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬────────────┘
            │ fetch + cache           │ read                    │ snapshot
            ▼                         ▼                         ▼
 ┌─────────────────────────────────────────────────┐  ┌──────────────────────┐
 │ Rankings parser + validator                     │  │ Schedule + results   │
 │ 32 teams, ranks 1..32 each used once;           │  │ loader               │
 │ NFL.com also: canonical URL, logo order         │  │ (REG season only)    │
 └────────────────────────┬────────────────────────┘  └───┬──────────────┬───┘
                          │ ranks                matchups │              │
                          ▼                               │              │
 ┌─────────────────────────────────────────────────┐      │              │
 │ Environment (one NFL week per step)             │◄─────┘              │
 │ percept = ranks + matchups (scores hidden)      │                     │
 └────────────────────────┬────────────────────────┘                     │
                          │ perceive                    observe: scores  │
                          ▼                     of completed games only  ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ PredictorAgent                                                            │
 │                                                                           │
 │  Feature registry ────────────► design matrix ──► Bayesian logistic model │
 │   rank_diff: NFL.com ranks, or yours            decide: P(home) from the  │
 │     (second forecast, for comparison)             posterior predictive    │
 │   point_diff, ppg_diff: computed from           update: MAP + Laplace on  │
 │     completed games only                          all completed games     │
 └──────────┬────────────────────────────────────────────────────────────────┘
            │ act: picks, probabilities, metrics report
            ▼
 ┌───────────────────────────┐     ┌─────────────────────────────────────────┐
 │ forecast_*.md             │     │ Evaluator                               │
 │ metrics_report_*.md       │────►│ accuracy, Brier, log loss, calibration  │
 │ trace_backtest.jsonl      │     │ ablation: Brier change per metric       │
 │ predictions_*.csv         │     │ your ranks vs. NFL.com, same games      │
 └───────────────────────────┘     │ baselines: home / rank / Vegas          │
                                   └─────────────────────────────────────────┘
```

The agent sees only the current week's rankings and matchups, plus results of games already
completed. Point differential and points per game are computed only from completed games, so
the current week's scores never enter a prediction. The user's rankings feed the same validator
as NFL.com's and drive a second forecast that is scored head to head (§7).

## 6. Schedule

Three weeks of implementation and evaluation, with one new metric each week, then a final week
for documentation and presentation. Each metric week follows the same loop: add the metric, run the
backtest with and without it, decide whether to keep it, publish the weights, and score the live
NFL week. NFL 2026 dates: week 9 is Nov 5–9, week 10 Nov 12–16, week 11 Nov 19–23, week 12
Nov 26–30 and week 13 Dec 3–7. Every live forecast is committed before that week's first kickoff.

| Project week | Metric added | Work | Decision | Live NFL weeks | Deliverable |
|---|---|---|---|---|---|
| **Week 1** (to Nov 5) | **NFL.com rank** (plus home field) | Proof of concept: rankings scraper and validator, Bayesian agent with one feature (`rank_diff`), walk-forward backtest for 2023–2026. | Is the approach feasible (§9)? | Forecast wk 9 | This document; `results/` evidence |
| **Week 2** (to Nov 12) | **Point differential** to date | Build the ablation harness: backtest with and without a metric, and report the change in Brier score with a bootstrap 95% CI and the metric's correlation with metrics already in the model. Add the weekly metrics report (weight ± sd for each metric). | Keep point differential if the CI for its Brier change is entirely an improvement. Otherwise drop it and report why. | Score wk 9; forecast wk 10 | Ablation table: rank vs. rank + point differential; metrics report |
| **Week 3** (to Nov 19) | **Points per game** to date | Same ablation with points per game. It overlaps heavily with point differential, so this also tests whether the ablation can separate correlated metrics (§4). **User rankings:** read `data/my_rankings.csv` and forecast each ranked week twice, once from NFL.com's ranks and once from the user's, with the same learned rank weight. Calibration plots. | Keep or drop points per game by the same rule. | Score wk 10; forecast wk 11 (both rank sources) | Full evaluation results; your rankings vs. NFL.com |
| Thanksgiving week (to Nov 26) | none | No project submission. Keep scoring live forecasts. | none | Score wk 11; forecast wk 12 | none |
| **Final week** (to Dec 3) | none | Limited refinement only. Final report, deploy/revise/do-not-use verdict against §10, slides. | Deploy, revise or do not use | Score wk 12; forecast wk 13 | Report and presentation |

## 7. Evaluation plan

**Protocol (repeatable).** Walk-forward replay. For every week, the agent sees only the rankings
published before that week and the results of earlier games. Ties are excluded from scoring. All
inputs are committed (`data/power_rankings_nflcom.csv`, `data/my_rankings.csv`, `data/games.csv`
snapshot), so `python -m nflpredictor backtest` reproduces every number in `results/summary.md`
exactly, including the bootstrap intervals. Those resample games with replacement, 2,000 times,
with a fixed seed (0), and report 95% percentile intervals.

**Live scoring procedure.** Weeks still to be played in 2026 form a true out-of-sample test.
1. Before the week's first kickoff, generate the forecast and commit it. The commit timestamp
   shows the forecast came first. The user's rankings for that week are committed the same way.
2. After the week's games, refresh `data/games.csv` and commit the new snapshot.
3. Score the forecast that was committed in step 1. It is never regenerated, even if the code
   has changed since.

**Numerical measures.**

| Property | Measure |
|---|---|
| Solution quality | Accuracy; Brier score; log loss, per season and pooled; bootstrap 95% confidence intervals for differences from baselines |
| Calibration | Observed win rate in each confidence bin; expected calibration error |
| Value of each metric | Change in Brier score when a feature is added (ablation), with bootstrap 95% CI; posterior weight ± sd; correlation with the metrics already in the model |
| User rankings vs. NFL.com | Brier score and accuracy of the two forecasts on the same live games, and the paired difference with a bootstrap CI |
| Reliability | Share of published NFL.com weeks parsed and validated; share of the user's ranking weeks that pass the same check (32 teams, ranks 1–32 each used once, committed before kickoff); share of games given a prediction; number of abstentions |
| Cost | Runtime per backtest; HTTP requests per week (1 per ranking source) |

**Baselines.** Home team always (p = 0.55); higher NFL.com rank always; and, as the ceiling,
the Vegas closing spread converted to a probability with Φ(spread / 13.5).

**Numbers used for the final verdict (§10).** Pooled scores over 2023–2025 and all live 2026 weeks,
with the live weeks also reported separately. Per-season scores are reported for context but do
not decide the verdict.

## 8. Proof of concept

### Run instructions

```bash
git clone [repository URL] nflpredictor && cd nflpredictor
git checkout [commit]
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m nflpredictor backtest                       # reproduces results/summary.md
.venv/bin/python -m nflpredictor forecast --season 2026 --week 9 # results/forecast_2026_w09.md
.venv/bin/python -m unittest discover tests                     # 5 tests (+1 parser test after fetch)
# optional: re-scrape the rankings (about 70 requests, 1 s apart)
.venv/bin/python -m nflpredictor fetch --seasons 2023 2024 2025 2026
```

Tested with Python 3.9.6, numpy 2.0.2 and pandas 2.3.3 on macOS. A full backtest takes about 0.1 s.

### What the proof of concept does

- **Input / initial state:** the 63 published NFL.com Power Rankings articles for 2023 week 1
  through 2026 week 9 (all 63 parsed and validated: 2,016 team-week ranks), the nflverse
  schedule, and an agent holding only its prior.
- **Strategy executed:** 72 weekly perceive–decide–act–observe–update steps. The agent made 952
  decisions and abstained in 9 weeks (2026 weeks 10–18, which have no rankings yet).
- **Observable result:** per-game picks and probabilities, a scored backtest, and a live forecast
  for 2026 week 9 (games Nov 5–9), generated on Nov 3, before kickoff.

### Retained evidence (in `results/`)

| File | Contents |
|---|---|
| `summary.md` | Score table, calibration table, final posterior, runtime |
| `trace_backtest.jsonl` | One line per week: prior, every decision (matchup, ranks, p_home, pick, final score), posterior after the update |
| `predictions_backtest.csv` | All 952 decisions, flat |
| `forecast_2026_w09.md` | Picks for the not-yet-played 2026 week 9 |
| `backtest_stdout.txt` | Console output of the backtest run |

**Backtest results (935 completed games, 2023 – 2026 week 8):**

| Method | Accuracy | Brier ↓ | Log loss ↓ |
|---|---|---|---|
| **Agent (NFL.com rank, Bayesian)** | **0.647** | **0.223** | **0.637** |
| Higher NFL.com rank always | 0.646 | (0.231)* | (0.654)* |
| Home team always | 0.539 | 0.249 | 0.690 |
| Vegas closing spread (ceiling) | 0.679 | 0.213 | 0.614 |

\*fixed 0.6/0.4 probability, included only so that the score is defined.

Agent accuracy 95% bootstrap CI: 0.615–0.678. Agent minus Vegas Brier score: +0.010 (95% CI
+0.005 to +0.016). Agent minus higher-rank rule: −0.008 (95% CI −0.013 to −0.002). All three
are generated by `backtest` and written to `summary.md`.

**Calibration:** in the 0.6–0.7 confidence bin, the agent's picks won 68.7% of the time at a mean stated
confidence of 64.6%. In the 0.8+ bin, they won 83.3% at a mean of 82.6%. Expected calibration
error is 0.031. As a check, I simulated 2,000 seasons in which the agent's own probabilities were
exactly right. Even then, these 935 games give a median calibration error of 0.022, and 0.040 at
the 95th percentile. So 0.031 is consistent with good calibration plus sampling noise.

**Learning visible in the trace:** the 2023 week 1 prior has β_rank = 0.00 ± 1.00 (the agent picks home
in every game). After week 1 it is 0.35 ± 0.37. By the end it is 0.558 ± 0.058, with home field
0.165 ± 0.069. Being ranked 10 spots higher is worth about +0.56 logits, roughly +13 to 14
percentage points of win probability.

**Sample from the trace (2023 week 1, prior only):**
`{"game": "DAL@NYG", "ranks": "6@19", "p_home": 0.528, "pick": "NYG", "final": "40-0"}`.
The agent hasn't yet learned that rank matters, which is the expected starting behavior.

## 9. Feasibility assessment

**Feasible. Proceed with the planned direction.** Both technical uncertainties are resolved:

1. **Sensing works.** NFL.com publishes every week at a predictable URL back to at least 2023.
   The parser reads the article body and checks it three ways (exactly 32 unique teams with ranks
   1–32, the canonical URL, and the order of the logos the page preloads). All 63 published
   weeks passed. The first draft of the checks flagged 2 pages incorrectly, and both were
   inspected and fixed. So when markup changes, the agent abstains instead of using bad data.
2. **The signal is real but limited.** Rankings alone beat picking the home team by 11
   points of accuracy and 0.026 in Brier score. The agent is well calibrated, which a bare ranking
   cannot be. But its accuracy matches the simple higher-rank rule, and its Brier score trails
   Vegas by a significant 0.010.

**What must change next.**
- Additional metrics are required, not optional. Rank alone is close to its ceiling. Rank order
  also loses margin information (#1 vs. #2 is not the same gap as #15 vs. #16), so point
  differential (Nov 12) and points per game (Nov 19) are the next additions.
- 2026 is the weakest season so far (57.5% on 120 games versus 60–69% in other seasons). Some of
  this is small-sample noise, but it coincides with a change of author: Eric Edholm wrote all 54
  rankings from 2023 to 2025, and Nick Shook writes them in 2026. The agent learned what one
  writer's rank is worth, and that may not carry over to another. Weighting recent games more
  heavily would address this, but for now every game counts equally to keep the agent simple.
  This remains an open issue (§10), and the 2026 rolling Brier score will show whether it matters.
- The agent picks the home team in 58% of games while home teams won 54%. This should be
  re-checked after more features are added.

## 10. Limitations, failure conditions, and recommendation criteria

**Expected limitations.**
- Rankings are published early in the week and miss later injury, quarterback and weather news.
- Each season provides only about 270 games, so differences of about 2 points of accuracy are
  within noise. Conclusions rely on Brier score, bootstrap intervals and multiple seasons.
- The model is linear in its features, with one home-field term for all teams.
- The market line uses far more information. Beating it is not an expected outcome.
- Every game counts equally, from 2023 to the most recent week. If what a ranking or stat is
  worth changes over time (for example, a new ranking writer), the agent adapts slowly.
- The user's own rankings can only be tested going forward, because past seasons can't be ranked
  without hindsight. By Dec 3 that is a few live weeks (about 14 games per week), so the
  comparison with NFL.com will be suggestive rather than conclusive.

**Failure conditions.**
- NFL.com changes its markup or URL scheme. Validation fails and the agent abstains. Detected,
  not silent.
- Ranking articles are late or missing. The agent abstains for that week.
- The ranking author or style changes (as happened in 2026). Calibration drifts, which shows in
  rolling Brier score.
- Team relocations or renames require updating the name map. The validator rejects the page
  until fixed.
- Week 1 relies on prior seasons only. Uncertainty is highest there.
- The user's rankings file is missing, invalid, or committed after kickoff. The agent skips the
  comparison forecast for that week and logs the reason. It never fills in or guesses ranks.
- The ablation can't separate correlated metrics: the Brier-change CI includes zero for both
  point differential and points per game. The metrics report then says these metrics carry the
  same information, rather than claiming either one matters on its own.

**Criteria for the final recommendation.** *Deploy* means publishing the weekly forecast and
metrics report for fans before each week's games, as a prediction and metric-analysis tool, not
for wagering. The verdict uses pooled scores over 2023–2025 plus all live 2026 weeks (§7).

| Verdict | Criteria |
|---|---|
| **Deploy** | All of: pooled Brier ≤ 0.218 (within 0.005 of Vegas); accuracy ≥ 66%; expected calibration error ≤ 0.04, the 95th percentile for a perfectly calibrated forecaster on the same games (§8); live 2026 Brier inside or below the backtest's bootstrap 95% CI; every kept metric's ablation CI shows a Brier improvement, and its weight has the same sign in every season; ≥ 98% of published weeks parsed and validated |
| **Revise** | Beats both heuristics on Brier with a CI that excludes zero, but misses one or more Deploy thresholds |
| **Do not use** | Fails to beat the higher-rank rule on Brier, is poorly calibrated (expected calibration error > 0.05), or data collection is unreliable |
| **Never** | Recommended for real-money betting unless it beats the closing line out of sample, which is not expected |

The comparison of the user's rankings with NFL.com's is reported as a finding but does not
affect the verdict. With about 14 games per live week, it is too small a sample to decide on.
