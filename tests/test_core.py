"""Run with: python -m unittest discover tests"""
import unittest

import numpy as np
import pandas as pd

from nflpredictor.agent import PredictorAgent, attach_rankings
from nflpredictor.data import RAW, TEAMS, parse_rankings, tier_ranks
from nflpredictor.evaluate import ats_picks, bootstrap, calibration, expected_calibration_error
from nflpredictor.features import nw_rank_diff
from nflpredictor.llm import LLMForecaster, parse, prompt
from nflpredictor.model import BayesLogit


class ModelTest(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(5000), rng.normal(size=5000)])
        y = (rng.random(5000) < 1 / (1 + np.exp(-(X @ [0.2, 0.8])))).astype(float)
        m = BayesLogit([0, 0], [1, 1]).fit(X, y)
        np.testing.assert_allclose(m.mean, [0.2, 0.8], atol=0.1)

    def test_predictions_are_probabilities(self):
        m = BayesLogit([0.15, 0.0], [0.5, 1.0])
        p = m.predict(np.array([[1.0, 3.0], [1.0, -3.0]]))
        self.assertTrue(np.all((p > 0) & (p < 1)))


class AgentTest(unittest.TestCase):
    def test_no_lookahead(self):
        """Decisions for a week must not change when that week's results change."""
        games = pd.DataFrame({
            "game_id": ["a", "b"], "season": [2026, 2026], "week": [1, 1],
            "away_team": ["BUF", "NYJ"], "home_team": ["MIA", "KC"],
            "away_score": [30, 10], "home_score": [10, 30], "result": [-20, 20], "spread_line": [-3, 7]})
        ranks = pd.DataFrame({"season": [2026] * 4, "week": [1] * 4,
                              "team": ["BUF", "MIA", "NYJ", "KC"], "rank": [1, 20, 30, 5]})
        env = attach_rankings(games, ranks)
        p1 = PredictorAgent(["rank_diff"]).decide(env).p_home.to_numpy()
        env2 = env.assign(result=-env.result)
        p2 = PredictorAgent(["rank_diff"]).decide(env2).p_home.to_numpy()
        np.testing.assert_array_equal(p1, p2)


class NickTiersTest(unittest.TestCase):
    def _week(self, tiers, names=""):
        return pd.DataFrame({"season": 2026, "week": 9, "tier": tiers, "tier_name": names,
                             "team": sorted(TEAMS.values())})

    def test_teams_in_a_tier_share_its_best_rank(self):
        tiers = [1, 2, 2, 3, 3, 3] + [4] * 26
        ranks = tier_ranks(self._week(tiers)).sort_values("team")["rank"].tolist()
        self.assertEqual(ranks, [1, 2, 2, 4, 4, 4] + [7] * 26)  # unnamed last tier: best rank

    def test_eliminated_tier_gets_average_rank(self):
        names = ["Contenders"] * 28 + ["Eliminated"] * 4
        ranks = tier_ranks(self._week([1] * 28 + [2] * 4, names)).sort_values("team")["rank"].tolist()
        self.assertEqual(ranks, [1] * 28 + [30.5] * 4)  # ranks 29-32 averaged

    def test_rejects_week_missing_a_team(self):
        with self.assertRaises(ValueError):
            tier_ranks(self._week([1] * 32).iloc[1:])

    def test_rejects_unfilled_tier(self):
        with self.assertRaises(ValueError):
            tier_ranks(self._week([1] * 31 + [None]))

    def test_feature_is_zero_without_tiers(self):
        games = pd.DataFrame({"home_nw_rank": [1, None], "away_nw_rank": [21, None]})
        np.testing.assert_allclose(nw_rank_diff(games), [2.0, 0.0])


class FakeModel:
    name = "fake"

    def __init__(self, replies):
        self.replies, self.temps, self.prompts = list(replies), [], []

    def complete(self, messages, temperature):
        self.temps.append(temperature)
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


class LLMTest(unittest.TestCase):
    game = pd.DataFrame({"game_id": ["g"], "season": [2026], "week": [9], "gameday": ["2026-11-05"],
                         "away_team": ["JAX"], "home_team": ["BAL"],
                         "away_rank": [4], "home_rank": [22], "away_record": ["6-2"], "home_record": ["3-5"],
                         "away_nw_rank": [3.0], "home_nw_rank": [23.0],
                         "away_nw_tier": ["That one Meme"], "home_nw_tier": ["My Name is Marshall"]})

    def test_blind_prompt_hides_the_chosen_inputs(self):
        text = prompt(self.game.iloc[0], "blind")
        self.assertIn("Jacksonville Jaguars", text)
        for leak in ("#4", "6-2", "Nick Wright", "Marshall", "Power Rankings"):
            self.assertNotIn(leak, text)

    def test_informed_prompt_includes_rank_record_and_tier(self):
        text = prompt(self.game.iloc[0], "informed")
        for item in ("JAX #4 (6-2)", "BAL #22 (3-5)", '"That one Meme" (rank 3)', "(rank 23)"):
            self.assertIn(item, text)

    def test_parse_strips_reasoning_and_follows_probability(self):
        out = parse('<think>hmm {"x": 1}</think> {"pick": "BAL", "p_home_win": 0.3, "reason": "r"}',
                    "JAX", "BAL")
        self.assertEqual((out["pick"], out["p_home"], out["pick_disagreed"]), ("JAX", 0.3, True))
        self.assertEqual(parse('{"pick": "JAX", "p_home_win": 65}', "JAX", "BAL")["p_home"], 0.65)

    def test_retries_then_abstains_and_passes_temperature(self):
        m = FakeModel(["no json here", '{"p_home_win": 150}', "still nothing"])
        d, trace = LLMForecaster(m, "blind", temperature=1.4).decide(self.game)
        self.assertTrue(np.isnan(d.p_home[0]))
        self.assertEqual(m.temps, [1.4, 1.4, 1.4])
        self.assertEqual(len(trace[0]["replies"]), 6)  # 3 replies + 3 parse errors

    def test_rejects_out_of_range_temperature(self):
        with self.assertRaises(ValueError):
            LLMForecaster(FakeModel([]), "blind", temperature=3)


class EvaluateTest(unittest.TestCase):
    def _decisions(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.3, 0.9, 20000)
        result = np.where(rng.random(20000) < p, 7, -7)
        return pd.DataFrame({"p_home": p, "confidence": np.maximum(p, 1 - p), "result": result,
                             "home_rank": 1, "away_rank": 2, "spread_line": 3.0})

    def test_bootstrap_is_repeatable(self):
        d = self._decisions()
        pd.testing.assert_frame_equal(bootstrap(d, n=50), bootstrap(d, n=50))

    def test_calibrated_forecasts_have_small_error(self):
        self.assertLess(expected_calibration_error(calibration(self._decisions())), 0.05)

    def test_ats_converts_confidence_to_points(self):
        d = pd.DataFrame({"p_home": [0.70, 0.70, 0.40, 0.60], "home_team": ["KC", "SF", "MIA", "NE"],
                          "away_team": ["NYJ", "LV", "DET", "GB"], "spread_line": [3.0, 7.0, -3.0, 2.0]})
        a = ats_picks(d)
        np.testing.assert_allclose(a.pred_margin, [4.0, 4.0, -2.0, 2.0])  # 1 point per 5 pts over 50%
        self.assertEqual(a.ats_pick.tolist(), ["KC", "LV", "MIA", None])  # equal to the line: no pick


@unittest.skipUnless((RAW / "2026_w09.html").exists(), "run `python -m nflpredictor fetch` first")
class ParserTest(unittest.TestCase):
    def test_week9_2026(self):
        df = parse_rankings((RAW / "2026_w09.html").read_text(), 2026, 9)
        self.assertEqual(df.team.tolist()[:3], ["BUF", "SF", "CHI"])  # matches the headline


if __name__ == "__main__":
    unittest.main()
