"""
train_rl.py
-----------
Standalone script to train the ADAPT Phase 2 Strategy Engine (PPO).
Loads a local bargaining dataset from the 'data/' directory.
"""

import os, glob, collections
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_EMOTIONS        = 7
EMOTION_NAMES       = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

RL_MAX_TURNS        = 25
RL_TOTAL_TIMESTEPS  = 80_000
RL_EVAL_FREQ        = 8_000
RL_MODEL_PATH       = "./phase2_brain"  # Saves to root directory
RL_LEARNING_RATE    = 0.0003
RL_BATCH_SIZE       = 128
RL_N_STEPS          = 2048

# Walk-away trigger thresholds
WA_INSULT_RATIO     = 0.30
WA_REPEAT_N         = 3
WA_NONSENSE_N       = 4
WA_PATIENCE_START   = 12
WA_NEGATIVE_N       = 3
WA_STALL_N          = 5

# Reward weights
RW_PROFIT, RW_DEAL_BONUS, RW_EFFICIENCY = 2.5, 1.5, 0.6
RW_WALKAWAY_GOOD, RW_WALKAWAY_BAD = 1.2, -2.5
RW_EARLY_DISC, RW_ANGER_CAVE = -0.6, -0.5
RW_BLUFF_DEFENSE, RW_GULLIBLE, RW_INSULT_DEFEND = 1.8, -2.2, -1.0

# ─── Reward Calculator ─────────────────────────────────────────────────────────
class RewardCalculator:
    def calculate(self, deal_made, walked_away, walk_was_justified,
                  final_price, listing_price, min_acceptable,
                  turn, max_turns, action, emotion_vector, bluff_prob,
                  text_sentiment_gap):
        c = {}

        if deal_made:
            price_range  = listing_price - min_acceptable + 1e-8
            profit_ratio = (final_price - min_acceptable) / price_range
            c['profit']  = RW_PROFIT * max(0, profit_ratio)
            c['deal']    = RW_DEAL_BONUS
            c['eff']     = RW_EFFICIENCY * (1.0 - turn / max_turns)
        elif walked_away:
            c['walk_good'] = RW_WALKAWAY_GOOD if walk_was_justified else RW_WALKAWAY_BAD
        elif turn >= max_turns:
            c['timeout'] = -1.5
        else:
            c['step_cost'] = -0.05

        if action in [1, 2] and turn <= 3:
            c['early_disc'] = RW_EARLY_DISC

        negative_level = float(emotion_vector[0] + emotion_vector[1]) if len(emotion_vector) > 1 else 0
        combined_bluff = max(bluff_prob, text_sentiment_gap)

        if combined_bluff > 0.55:
            if action == 5:           c['bluff_def']  = RW_BLUFF_DEFENSE
            elif action in [1, 2]:    c['gullible']   = RW_GULLIBLE
        else:
            if negative_level > 0.5 and action == 5:
                c['insult']     = RW_INSULT_DEFEND
            elif negative_level > 0.5 and action in [1, 2]:
                c['anger_cave'] = RW_ANGER_CAVE

        return sum(c.values()), c

# ─── Gymnasium Environment ─────────────────────────────────────────────────────
class NegotiationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset_df=None):
        super().__init__()
        self.action_space      = spaces.Discrete(7)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(24,), dtype=np.float32)
        self.max_turns         = RL_MAX_TURNS
        self.reward_calc       = RewardCalculator()
        self.dataset           = dataset_df
        self.reset()

    def _simulate_buyer_state(self):
        emotion    = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        bluff_prob = 0.0
        text_gap   = 0.0
        gap = (self.seller_price - self.buyer_price) / max(self.listing_price, 1e-8)

        if gap > 0.30:
            emotion[0] = 0.55 + self.buyer_aggression * 0.35
            if np.random.rand() > 0.65:
                bluff_prob, text_gap = 1.0, np.random.uniform(0.6, 0.95)
                emotion[3], emotion[4], emotion[0] = 0.45, 0.35, 0.15
            else:
                text_gap = np.random.uniform(0.0, 0.15)
        elif gap > 0.18:
            emotion[5], emotion[4], text_gap = 0.30, 0.45, np.random.uniform(0.0, 0.20)
        elif gap > 0.08:
            emotion[3], emotion[4], text_gap = 0.50, 0.30, np.random.uniform(0.0, 0.10)
        else:
            emotion[3], emotion[6], text_gap = 0.75, 0.15, np.random.uniform(0.0, 0.05)

        if self.turn > self.max_turns * 0.70:
            emotion[0] = min(emotion[0] + 0.20, 1.0)
            emotion[5] = min(emotion[5] + 0.10, 1.0)
        if self.negative_streak >= WA_NEGATIVE_N - 1:
            emotion[0] = min(emotion[0] + 0.30, 1.0)

        total  = emotion.sum()
        return (emotion / total) if total > 0 else np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), bluff_prob, text_gap

    def _get_observation(self):
        norm = lambda x: float(np.clip(x / max(self.listing_price, 1e-8), 0, 1))
        emotion, bluff_prob, text_gap = self._simulate_buyer_state()
        self.current_emotion, self.current_bluff, self.current_text_gap = emotion, bluff_prob, text_gap

        padded = np.zeros(5, dtype=np.float32)
        hist   = [norm(p) for p in self.price_history[-5:]]
        padded[:len(hist)] = hist

        obs = np.concatenate([
            [norm(self.seller_price)], [self.turn / self.max_turns], emotion, padded,
            [norm(self.seller_price - self.buyer_price)], [self.buyer_patience],
            [self.buyer_aggression], [min(self.rounds_since_concession / 5.0, 1.0)],
            [min(self.total_concessions / 6.0, 1.0)], [bluff_prob],
            [min(self.nonsense_streak / WA_NONSENSE_N, 1.0)], [min(self.stall_turns / WA_STALL_N, 1.0)],
            [min(self.negative_streak / WA_NEGATIVE_N, 1.0)], [self.product_value_score]
        ]).astype(np.float32)
        return obs

    def _check_walkaway_justified(self) -> bool:
        if self.buyer_price < self.listing_price * WA_INSULT_RATIO: return True
        if self.nonsense_streak >= WA_NONSENSE_N: return True
        if self.stall_turns >= WA_STALL_N: return True
        if self.negative_streak >= WA_NEGATIVE_N: return True
        if self.buyer_patience <= 0.05: return True
        hist = list(self.offer_history)
        if len(hist) >= WA_REPEAT_N and len(set(round(x, 0) for x in hist[-WA_REPEAT_N:])) == 1: return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.dataset is not None and not self.dataset.empty:
            row = self.dataset.sample(1).iloc[0]
            price_col = next((c for c in row.index if c.lower() == 'price'), None)
            if price_col and pd.notna(row[price_col]):
                try:
                    raw = str(row[price_col]).replace('$', '').replace(',', '').strip()
                    self.listing_price = max(float(raw), 10.0)
                except Exception:
                    self.listing_price = float(np.random.choice([np.random.uniform(10, 100), np.random.uniform(100, 500), np.random.uniform(500, 5000)]))
            else:
                self.listing_price = float(np.random.choice([np.random.uniform(10, 100), np.random.uniform(100, 500), np.random.uniform(500, 5000)]))

            title_col = next((c for c in row.index if c.lower() == 'title'), None)
            if title_col and pd.notna(row[title_col]):
                self.product_name = str(row[title_col])
            elif 'class_name' in row and pd.notna(row['class_name']):
                self.product_name = str(row['class_name']).capitalize()
            else:
                self.product_name = "Item"
        else:
            self.listing_price = float(np.random.choice([np.random.uniform(10, 100), np.random.uniform(100, 500), np.random.uniform(500, 5000)]))
            self.product_name = "Item"

        self.product_value_score = float(np.clip(np.log10(max(self.listing_price, 1)) / np.log10(5000), 0, 1))
        self.min_acceptable  = self.listing_price * 0.68
        self.buyer_target    = self.listing_price * np.random.uniform(0.40, 0.65)

        self.buyer_patience    = np.random.uniform(0.30, 1.0)
        self.buyer_aggression  = np.random.uniform(0.00, 0.85)
        self.buyer_willingness = np.random.uniform(0.35, 0.95)

        self.seller_price = self.listing_price
        self.buyer_price = self.buyer_target
        self.turn, self.rounds_since_concession, self.total_concessions = 0, 0, 0
        self.nonsense_streak, self.stall_turns, self.negative_streak = 0, 0, 0
        self.price_history = [self.seller_price]
        self.offer_history = collections.deque(maxlen=WA_REPEAT_N)
        self.last_buyer_price = self.buyer_price
        self.current_emotion = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        self.current_bluff, self.current_text_gap = 0.0, 0.0
        self.done, self.deal_made, self.walked_away = False, False, False

        return self._get_observation(), {}

    def step(self, action):
        if self.done: return self._get_observation(), 0.0, True, False, {}

        self.turn += 1
        prev_buyer_price, walk_justified = self.buyer_price, False

        if action == 0: self.rounds_since_concession += 1
        elif action == 1:
            self.seller_price -= np.random.uniform(0.02, 0.05) * self.seller_price
            self.total_concessions += 1; self.rounds_since_concession = 0
        elif action == 2:
            self.seller_price -= np.random.uniform(0.05, 0.10) * self.seller_price
            self.total_concessions += 1; self.rounds_since_concession = 0
        elif action == 3:
            self.seller_price -= np.random.uniform(0.01, 0.03) * self.seller_price
            self.total_concessions += 1; self.rounds_since_concession = 0
        elif action == 4:
            if self.seller_price >= self.min_acceptable: self.deal_made = True; self.done = True
        elif action == 5: self.rounds_since_concession += 1
        elif action == 6:
            walk_justified, self.walked_away, self.done = self._check_walkaway_justified(), True, True

        self.seller_price = max(self.seller_price, self.min_acceptable)
        self.price_history.append(self.seller_price)

        if not self.done:
            gap = self.seller_price - self.buyer_price
            concession_rate = 0.04 + 0.12 * self.buyer_willingness
            if action == 5 and self.current_bluff > 0.55: concession_rate *= 2.2
            if self.rounds_since_concession >= 3: concession_rate *= 0.45
            if action == 5 and self.current_bluff < 0.30: self.buyer_patience -= 0.08

            self.buyer_price += gap * concession_rate
            self.offer_history.append(self.buyer_price)

            self.stall_turns = self.stall_turns + 1 if abs(self.buyer_price - prev_buyer_price) < self.listing_price * 0.005 else 0
            self.negative_streak = self.negative_streak + 1 if self.current_emotion[0] > 0.35 else 0
            self.nonsense_streak = self.nonsense_streak + 1 if self.current_bluff > 0.8 and gap / max(self.listing_price, 1) > 0.35 else 0

            if self.buyer_price >= self.seller_price: self.deal_made = True; self.done = True

            drain = 0.04 + 0.05 * self.buyer_aggression
            if action == 5 and self.current_bluff < 0.30: drain *= 2.5
            self.buyer_patience -= drain

            if self.buyer_patience <= 0 or self.turn >= self.max_turns: self.done = True

        reward, components = self.reward_calc.calculate(
            self.deal_made, self.walked_away, walk_justified, self.seller_price, self.listing_price,
            self.min_acceptable, self.turn, self.max_turns, action, self.current_emotion, self.current_bluff, self.current_text_gap
        )

        return self._get_observation(), reward, self.done, False, {'deal_made': self.deal_made, 'walked_away': self.walked_away, 'walk_justified': walk_justified, 'final_price': self.seller_price, 'listing_price': self.listing_price, 'reward_components': components}

# ─── Training ──────────────────────────────────────────────────────────────────
def load_local_data(data_dir="data"):
    if not os.path.exists(data_dir):
        print(f"⚠️ Data directory '{data_dir}' not found. Using synthetic data.")
        return None
    
    files = glob.glob(f"{data_dir}/*.jsonl") + glob.glob(f"{data_dir}/*.csv")
    if not files:
        print(f"⚠️ No .jsonl or .csv files found in '{data_dir}'. Using synthetic data.")
        return None
        
    print(f"Loading dataset from {files[0]}...")
    try:
        if files[0].endswith('.jsonl'):
            return pd.read_json(files[0], lines=True)
        else:
            return pd.read_csv(files[0])
    except Exception as e:
        print(f"⚠️ Failed to load dataset: {e}. Using synthetic data.")
        return None

def train_negotiation_agent():
    print("=" * 70)
    print("ADAPT Phase 2 — Local PPO Brain Training")
    print("=" * 70)

    df = load_local_data("data")
    if df is not None:
        print(f"✅ Loaded local dataset ({len(df)} rows)")

    train_env = Monitor(NegotiationEnv(dataset_df=df))
    eval_env  = Monitor(NegotiationEnv(dataset_df=df))
    check_env(train_env.unwrapped)

    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        learning_rate = RL_LEARNING_RATE,
        batch_size    = RL_BATCH_SIZE,
        n_steps       = RL_N_STEPS,
        n_epochs      = 12,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.015,
        policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        device        = "cpu",
    )

    callbacks = [EvalCallback(
        eval_env,
        best_model_save_path = ".",
        eval_freq            = RL_EVAL_FREQ,
        n_eval_episodes      = 100,
        deterministic        = True,
    )]

    print(f"\nTraining for {RL_TOTAL_TIMESTEPS:,} timesteps…")
    model.learn(total_timesteps=RL_TOTAL_TIMESTEPS, callback=callbacks)
    model.save(RL_MODEL_PATH)
    print(f"\n✅ Model saved locally as {RL_MODEL_PATH}.zip")

if __name__ == "__main__":
    train_negotiation_agent()