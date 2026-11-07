# Backtest summary

Seasons: 2023, 2024, 2025, 2026. Features: rank_diff. Walk-forward: each week is predicted using only games completed before it.

## Scores (completed, non-tied games)

| season   | method                          |   games |   accuracy |   brier |   log_loss |
|:---------|:--------------------------------|--------:|-----------:|--------:|-----------:|
| 2023     | agent (rankings, Bayesian)      |     272 |      0.603 |   0.235 |      0.662 |
| 2023     | baseline: higher NFL.com rank   |     272 |      0.610 |   0.238 |      0.669 |
| 2023     | baseline: home team             |     272 |      0.555 |   0.247 |      0.687 |
| 2023     | reference: Vegas closing spread |     272 |      0.680 |   0.217 |      0.624 |
| 2024     | agent (rankings, Bayesian)      |     272 |      0.688 |   0.211 |      0.612 |
| 2024     | baseline: higher NFL.com rank   |     272 |      0.702 |   0.220 |      0.632 |
| 2024     | baseline: home team             |     272 |      0.533 |   0.249 |      0.692 |
| 2024     | reference: Vegas closing spread |     272 |      0.713 |   0.203 |      0.594 |
| 2025     | agent (rankings, Bayesian)      |     271 |      0.683 |   0.214 |      0.616 |
| 2025     | baseline: higher NFL.com rank   |     271 |      0.635 |   0.233 |      0.659 |
| 2025     | baseline: home team             |     271 |      0.539 |   0.249 |      0.690 |
| 2025     | reference: Vegas closing spread |     271 |      0.653 |   0.212 |      0.611 |
| 2026     | agent (rankings, Bayesian)      |     120 |      0.575 |   0.245 |      0.686 |
| 2026     | baseline: higher NFL.com rank   |     120 |      0.625 |   0.235 |      0.663 |
| 2026     | baseline: home team             |     120 |      0.517 |   0.251 |      0.695 |
| 2026     | reference: Vegas closing spread |     120 |      0.658 |   0.226 |      0.645 |
| all      | agent (rankings, Bayesian)      |     935 |      0.647 |   0.223 |      0.637 |
| all      | baseline: higher NFL.com rank   |     935 |      0.646 |   0.231 |      0.654 |
| all      | baseline: home team             |     935 |      0.539 |   0.249 |      0.690 |
| all      | reference: Vegas closing spread |     935 |      0.679 |   0.213 |      0.614 |

Note: the higher-rank baseline is scored on accuracy; its fixed 0.6/0.4 probability only makes Brier/log loss defined.

## Agent calibration (all seasons)

| bin          |   games |   mean_confidence |   win_rate |
|:-------------|--------:|------------------:|-----------:|
| (0.499, 0.6] |     418 |             0.549 |      0.574 |
| (0.6, 0.7]   |     307 |             0.646 |      0.687 |
| (0.7, 0.8]   |     162 |             0.740 |      0.704 |
| (0.8, 1.0]   |      48 |             0.826 |      0.833 |

Expected calibration error: 0.031

## Bootstrap 95% intervals (all seasons)

2,000 resamples of games with replacement, seed 0, percentile intervals. For Brier differences, negative means the agent is better.

| statistic                          |   estimate |   ci_low |   ci_high |
|:-----------------------------------|-----------:|---------:|----------:|
| agent accuracy                     |      0.647 |    0.615 |     0.678 |
| agent Brier                        |      0.223 |    0.214 |     0.233 |
| agent Brier − higher NFL.com rank  |     -0.008 |   -0.013 |    -0.002 |
| agent Brier − home team            |     -0.026 |   -0.034 |    -0.016 |
| agent Brier − Vegas closing spread |      0.010 |    0.005 |     0.016 |

## Against the spread

Rule: 1 point(s) of predicted margin per 5 percentage points of confidence above 50%; pick the side the predicted margin favors relative to the closing spread. Pushes and games without a line are excluded. Break-even at -110 odds is 52.4%.

| season   | method             |   games |   cover_rate |   ci_low |   ci_high |
|:---------|:-------------------|--------:|-------------:|---------:|----------:|
| 2023     | agent ATS pick     |     258 |        0.484 |  nan     |   nan     |
| 2023     | baseline: favorite |     258 |        0.531 |  nan     |   nan     |
| 2023     | baseline: underdog |     258 |        0.469 |  nan     |   nan     |
| 2024     | agent ATS pick     |     268 |        0.478 |  nan     |   nan     |
| 2024     | baseline: favorite |     268 |        0.534 |  nan     |   nan     |
| 2024     | baseline: underdog |     268 |        0.466 |  nan     |   nan     |
| 2025     | agent ATS pick     |     271 |        0.517 |  nan     |   nan     |
| 2025     | baseline: favorite |     271 |        0.483 |  nan     |   nan     |
| 2025     | baseline: underdog |     271 |        0.517 |  nan     |   nan     |
| 2026     | agent ATS pick     |     115 |        0.487 |  nan     |   nan     |
| 2026     | baseline: favorite |     115 |        0.487 |  nan     |   nan     |
| 2026     | baseline: underdog |     115 |        0.513 |  nan     |   nan     |
| all      | agent ATS pick     |     912 |        0.492 |    0.459 |     0.524 |
| all      | baseline: favorite |     912 |        0.512 |    0.478 |     0.543 |
| all      | baseline: underdog |     912 |        0.488 |    0.457 |     0.522 |

## Final posterior (logit scale)

- home_field: mean +0.165, sd 0.069
- rank_diff: mean +0.558, sd 0.058

Runtime: 0.92 s for 952 decisions. Trace: `results/trace_backtest.jsonl`.
