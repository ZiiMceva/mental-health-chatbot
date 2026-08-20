"""
LinUCB Contextual Bandit — Game Recommender (V2)
=================================================
Uses one-hot emotion encoding for better game selection.

Usage:
    from core.bandit import LinUCBAgent, GAME_NAMES
    agent = LinUCBAgent()
    vector = agent.make_vector(0.9, 'anxiety', 0.8, 0.7, True)
    game_idx = agent.select(vector)
    agent.update(game_idx, vector, reward)
    agent.save("bandit_weights.npz")
"""

import numpy as np
from pathlib import Path

GAME_NAMES = [
    "breathing",       # 0 — acute anxiety/panic
    "puzzle",          # 1 — cognitive overload
    "attention_tap",   # 2 — rumination
    "nature_walk",     # 3 — burnout/exhaustion
    "journal",         # 4 — sadness/suppressed emotion
]

EMOTION_MAP = {
    "anxiety":  0,
    "burnout":  1,
    "sadness":  2,
    "overwhelm":3,
    "neutral":  4,
}

NUM_ARMS  = len(GAME_NAMES)
STATE_DIM = 9  # 5 one-hot emotions + stress + cognitive_load + engagement + needs_game


class LinUCBAgent:
    """
    LinUCB with one-hot emotion encoding.
    State vector: [anxiety, burnout, sadness, overwhelm, neutral,
                   stress_level, cognitive_load, engagement, needs_game]
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.d     = STATE_DIM
        self.A     = [np.eye(self.d)   for _ in range(NUM_ARMS)]
        self.b     = [np.zeros(self.d) for _ in range(NUM_ARMS)]
        self.total_steps = 0

    def make_vector(
        self,
        stress_level:   float,
        emotion_type:   str,
        cognitive_load: float,
        engagement:     float,
        needs_game:     bool,
    ) -> list[float]:
        """
        Convert stress state to one-hot feature vector.
        Returns list of 9 floats.
        """
        one_hot = [0.0] * 5
        one_hot[EMOTION_MAP.get(emotion_type, 4)] = 1.0
        return one_hot + [
            float(stress_level),
            float(cognitive_load),
            float(engagement),
            1.0 if needs_game else 0.0,
        ]

    def select(self, vector: list[float]) -> int:
        """Select best game for given state vector."""
        x = np.array(vector, dtype=float)
        scores = []
        for arm in range(NUM_ARMS):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            ucb   = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            scores.append(ucb)
        return int(np.argmax(scores))

    def update(self, arm: int, vector: list[float], reward: float):
        """Update model after observing reward."""
        x = np.array(vector, dtype=float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.total_steps += 1

    def get_game_name(self, arm_idx: int) -> str:
        return GAME_NAMES[arm_idx]

    def get_probabilities(self, vector: list[float]) -> dict:
        """Return softmax-normalised scores for UI display."""
        x = np.array(vector, dtype=float)
        raw = []
        for arm in range(NUM_ARMS):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            raw.append(float(theta @ x))
        e    = np.exp(np.array(raw) - np.max(raw))
        probs = e / e.sum()
        return {GAME_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_ARMS)}

    def save(self, path: str):
        """Save weights to disk."""
        np.savez(
            path,
            alpha=np.array([self.alpha]),
            total_steps=np.array([self.total_steps]),
            **{f"A_{i}": self.A[i] for i in range(NUM_ARMS)},
            **{f"b_{i}": self.b[i] for i in range(NUM_ARMS)},
        )
        print(f"Saved bandit weights to {path}")

    @classmethod
    def load(cls, path: str) -> "LinUCBAgent":
        """Load saved weights."""
        data  = np.load(path)
        agent = cls(alpha=float(data["alpha"][0]))
        agent.total_steps = int(data["total_steps"][0])
        for i in range(NUM_ARMS):
            agent.A[i] = data[f"A_{i}"]
            agent.b[i] = data[f"b_{i}"]
        print(f"Loaded bandit weights from {path} "
              f"(steps={agent.total_steps})")
        return agent

    def pretrain(self, episodes_csv: str):
        """Pre-train on synthetic RL data."""
        import pandas as pd
        df = pd.read_csv(episodes_csv)
        print(f"Pre-training on {len(df):,} synthetic episodes...")
        for _, row in df.iterrows():
            vector = self.make_vector(
                stress_level=row["stress_level"],
                emotion_type=row["emotion_type"],
                cognitive_load=row["cognitive_load"],
                engagement=row["engagement"],
                needs_game=row["stress_level"] > 0.6,
            )
            self.update(int(row["action"]), vector, float(row["reward"]))
        print(f"Pre-training complete. Total updates: {self.total_steps:,}")


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    agent = LinUCBAgent(alpha=0.5)

    # Pretrain
    import os
    synthetic = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "data", "processed", "synthetic_rl", "synthetic_episodes_v2.csv"
    )
    if os.path.exists(synthetic):
        agent.pretrain(synthetic)
    
    # Test each emotion
    test_cases = [
        ("anxiety",   "anxiety",   0.9, 0.8, 0.7, True),
        ("burnout",   "burnout",   0.7, 0.4, 0.3, True),
        ("sadness",   "sadness",   0.6, 0.5, 0.5, True),
        ("overwhelm", "overwhelm", 0.8, 0.9, 0.6, True),
        ("neutral",   "neutral",   0.2, 0.3, 0.8, False),
    ]

    correct_map = {
        "anxiety":   "breathing",
        "burnout":   "nature_walk",
        "sadness":   "journal",
        "overwhelm": "breathing",
        "neutral":   "puzzle",
    }

    print(f"\n{'Emotion':<12} {'Got':<16} {'Expected':<16} {'OK?'}")
    print("-"*55)
    correct = 0
    for label, emotion, stress, cog, eng, needs in test_cases:
        vector   = agent.make_vector(stress, emotion, cog, eng, needs)
        game_idx = agent.select(vector)
        game     = agent.get_game_name(game_idx)
        ok       = game == correct_map[label]
        if ok:
            correct += 1
        print(f"{label:<12} {game:<16} {correct_map[label]:<16} "
              f"{'✅' if ok else '❌'}")

    print(f"\nAccuracy: {correct}/5 = {correct/5:.0%}")