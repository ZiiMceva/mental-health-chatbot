"""
Stress State Encoder
=====================
Sends conversation history to GPT-4o-mini and extracts a structured
stress state vector as JSON. This is Novel Contribution C1.

Usage:
    from core.stress_encoder import encode_stress
    state = encode_stress(conversation_history)
    # → {"stress_level": 0.78, "emotion_type": "anxiety", ...}
"""

import os, json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a clinical stress analyst embedded in a mental health support chatbot.
After reading the conversation, output ONLY a raw JSON object — no markdown,
no explanation, no extra text. Use EXACTLY this schema:

{
  "stress_level":    <float 0.0–1.0 — overall stress intensity>,
  "emotion_type":    <"anxiety" | "burnout" | "sadness" | "overwhelm" | "neutral">,
  "cognitive_load":  <float 0.0–1.0 — how mentally overloaded the user seems>,
  "engagement":      <float 0.0–1.0 — how actively/openly the user is communicating>,
  "needs_game":      <true | false — whether a game intervention is appropriate now>,
  "reasoning":       <string — one sentence explaining your assessment>
}

Rules:
- stress_level > 0.6 generally means a game is helpful (needs_game: true)
- emotion_type must be exactly one of the five options above
- Be conservative: if uncertain, lean toward lower stress_level
- engagement 0.0 = one-word replies, 1.0 = long expressive messages
"""


class StressState(BaseModel):
    stress_level:   float = Field(ge=0.0, le=1.0)
    emotion_type:   Literal["anxiety", "burnout", "sadness", "overwhelm", "neutral"]
    cognitive_load: float = Field(ge=0.0, le=1.0)
    engagement:     float = Field(ge=0.0, le=1.0)
    needs_game:     bool
    reasoning:      str

    def to_vector(self) -> list[float]:
        """Returns numeric feature vector for the RL bandit."""
        emotion_map = {"anxiety": 0, "burnout": 1, "sadness": 2,
                       "overwhelm": 3, "neutral": 4}
        return [
            self.stress_level,
            emotion_map[self.emotion_type] / 4.0,   # normalise to 0–1
            self.cognitive_load,
            self.engagement,
            1.0 if self.needs_game else 0.0,
        ]


def encode_stress(
    conversation_history: list[dict],
    model: str = "gpt-4o-mini",
) -> StressState:
    """
    Args:
        conversation_history: list of {"role": "user"/"assistant", "content": str}
        model: OpenAI model to use (gpt-4o-mini is cheap and fast enough)

    Returns:
        StressState pydantic model with validated fields
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,      # low temp → consistent JSON
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return StressState(**data)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_convo = [
        {"role": "user",      "content": "I can't sleep and my deadline is tomorrow. I feel like everything is falling apart."},
        {"role": "assistant", "content": "I'm sorry to hear that. Can you tell me more about what's been going on?"},
        {"role": "user",      "content": "Just too much work, I haven't eaten properly in two days and my head is pounding."},
    ]

    print("Encoding stress state...")
    state = encode_stress(test_convo)
    print(f"\nStress State:")
    print(f"  stress_level:  {state.stress_level}")
    print(f"  emotion_type:  {state.emotion_type}")
    print(f"  cognitive_load:{state.cognitive_load}")
    print(f"  engagement:    {state.engagement}")
    print(f"  needs_game:    {state.needs_game}")
    print(f"  reasoning:     {state.reasoning}")
    print(f"\nFeature vector: {state.to_vector()}")
