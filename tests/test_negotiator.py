"""
tests/test_negotiator.py
------------------------
Automated regression and integrity test suite for Project ADAPT.
Runs standalone via: python -m unittest discover -s tests
"""

import os
import sys
import unittest
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from train_rl import NegotiationEnv, RewardCalculator
from app import extract_price, _safe_vec, compute_bluff_score, RLNegotiationAgent, _NEUTRAL_VEC


class TestPriceExtraction(unittest.TestCase):
    """Verifies that price parsing handles commas, colloquial phrasing, and edge cases."""

    def test_comma_formatted_prices(self):
        self.assertEqual(extract_price("$1,200"), 1200.0)
        self.assertEqual(extract_price("I can offer $2,500.50"), 2500.50)
        self.assertEqual(extract_price("$10,000"), 10000.0)

    def test_standard_dollar_formats(self):
        self.assertEqual(extract_price("$150"), 150.0)
        self.assertEqual(extract_price("$75.25"), 75.25)
        self.assertEqual(extract_price("300$"), 300.0)

    def test_colloquial_phrasing(self):
        self.assertEqual(extract_price("I can do 200"), 200.0)
        self.assertEqual(extract_price("How about 180?"), 180.0)
        self.assertEqual(extract_price("Can you do 250 bucks?"), 250.0)
        self.assertEqual(extract_price("I will pay 120 dollars"), 120.0)
        self.assertEqual(extract_price("I can only offer 95"), 95.0)

    def test_non_price_inputs(self):
        self.assertIsNone(extract_price("Hello, is this still available?"))
        self.assertIsNone(extract_price("I have 2 dogs at home."))
        self.assertIsNone(extract_price(""))


class TestEmotionSensorsAndBluff(unittest.TestCase):
    """Verifies mathematical stability of emotion vector normalization and bluff calculation."""

    def test_safe_vec_normalization(self):
        vec = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
        safe = _safe_vec(vec)
        self.assertAlmostEqual(float(safe.sum()), 1.0, places=5)
        self.assertTrue(np.all(safe >= 0.0))

    def test_safe_vec_nan_and_zero_safety(self):
        nan_vec = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        safe = _safe_vec(nan_vec)
        self.assertFalse(np.isnan(safe).any())
        self.assertAlmostEqual(float(safe.sum()), 1.0, places=5)

    def test_bluff_score_bounds_and_logic(self):
        # Calm face (Happy + Neutral) + Highly hostile text (Angry + Disgust)
        calm_face = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0], dtype="float32")
        hostile_text = np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1], dtype="float32")
        score = compute_bluff_score(calm_face, hostile_text)
        self.assertGreater(score, 0.6)
        self.assertLessEqual(score, 1.0)

        # Consistent anger (angry face + angry text) -> genuine frustration, not bluff
        angry_face = np.array([0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0], dtype="float32")
        score2 = compute_bluff_score(angry_face, hostile_text)
        self.assertLess(score2, score)


class TestRLGymEnvironment(unittest.TestCase):
    """Verifies observation bounds and step dynamics of the RL environment."""

    def setUp(self):
        self.env = NegotiationEnv()

    def test_observation_space_bounds(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (24,))
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0),
                        f"Observation out of bounds: min={obs.min()}, max={obs.max()}")

    def test_step_cost_vs_timeout(self):
        calc = RewardCalculator()
        # Normal intermediate step (turn 2/25, no deal, no walkaway)
        _, comp = calc.calculate(
            deal_made=False, walked_away=False, walk_was_justified=False,
            final_price=180, listing_price=200, min_acceptable=136,
            turn=2, max_turns=25, action=0,
            emotion_vector=np.zeros(7), bluff_prob=0.0, text_sentiment_gap=0.0
        )
        self.assertIn('step_cost', comp)
        self.assertEqual(comp['step_cost'], -0.05)
        self.assertNotIn('timeout', comp)

        # Timeout at max_turns
        _, comp_timeout = calc.calculate(
            deal_made=False, walked_away=False, walk_was_justified=False,
            final_price=180, listing_price=200, min_acceptable=136,
            turn=25, max_turns=25, action=0,
            emotion_vector=np.zeros(7), bluff_prob=0.0, text_sentiment_gap=0.0
        )
        self.assertIn('timeout', comp_timeout)
        self.assertEqual(comp_timeout['timeout'], -1.5)


class TestRLNegotiationAgent(unittest.TestCase):
    """Verifies state machine lifecycle, walkaway triggers, and deal acceptance."""

    def setUp(self):
        # Initialize agent with fallback heuristic (rl_model=None) to test deterministic state machine
        self.agent = RLNegotiationAgent(rl_model=None)
        self.agent.reset(listing_price=200.0, min_acceptable=120.0, product_name="Headphones")

    def test_insult_offer_walkaway(self):
        # Offering $40 on a $200 item is 20% (< 30% insult threshold)
        res = self.agent.process_message(
            "I'll give you $40",
            np.array(_NEUTRAL_VEC, dtype="float32"),
            np.array(_NEUTRAL_VEC, dtype="float32"),
            bluff_score=0.1
        )
        self.assertTrue(res["walkaway"])
        self.assertEqual(res["action_name"], "walk_away")
        self.assertTrue(self.agent.state["negotiation_over"])

    def test_repeat_offer_walkaway(self):
        # Repeating identical offer 3 times in a row
        neutral = np.array(_NEUTRAL_VEC, dtype="float32")
        for _ in range(2):
            self.agent.process_message("How about $140?", neutral, neutral, 0.1)
            self.assertFalse(self.agent.state["negotiation_over"])

        # 3rd identical offer
        res = self.agent.process_message("How about $140?", neutral, neutral, 0.1)
        self.assertTrue(res["walkaway"])
        self.assertIn("Same offer repeated", res["reason"])

    def test_deal_acceptance_on_full_price(self):
        neutral = np.array(_NEUTRAL_VEC, dtype="float32")
        res = self.agent.process_message("I will pay $200", neutral, neutral, 0.0)
        self.assertTrue(res["deal_made"])
        self.assertEqual(res["action_name"], "accept_deal")
        self.assertTrue(self.agent.state["negotiation_over"])
        self.assertEqual(self.agent.state["seller_price"], 200.0)


if __name__ == "__main__":
    unittest.main()
