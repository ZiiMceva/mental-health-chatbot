# Mental Health Chatbot — RL-Driven Therapeutic Game Selection

> Usama · Sogang University M.S. Research Project · 2025

An intelligent mental health web application that detects user stress through conversation
using an LLM, then uses a Reinforcement Learning (LinUCB bandit) agent to recommend the
most appropriate therapeutic mini-game. The system learns which games work best for each
user's stress profile over time.

---

## Quick Start (3 commands)

```bash
git clone <your-repo-url> && cd mental-health-chatbot
python setup.py          # creates venv, installs deps, downloads all datasets
# Add OPENAI_API_KEY to backend/.env, then:
cd backend && source venv/bin/activate && uvicorn main:app --reload
```

---

## Project Structure

```
mental-health-chatbot/
│
├── setup.py                        ← Run this first — sets up everything
├── mental-health-chatbot.code-workspace ← Open this in VS Code
│
├── scripts/
│   └── download_datasets.py        ← Auto-downloads all 5 datasets + generates synthetic RL data
│
├── backend/                        ← FastAPI Python backend
│   ├── main.py                     ← App entry point
│   ├── requirements.txt            ← All Python dependencies
│   ├── .env.example                ← Copy to .env and fill in keys
│   │
│   ├── core/                       ← Novel ML components
│   │   ├── stress_encoder.py       ← [C1] LLM → structured stress state vector
│   │   ├── bandit.py               ← [C2] LinUCB RL agent — game selector
│   │   └── memory.py               ← ChromaDB RAG — cross-session memory (Week 4)
│   │
│   ├── api/                        ← FastAPI route handlers (added Week 2+)
│   │   ├── chat.py                 ← POST /api/chat — conversation endpoint
│   │   ├── games.py                ← POST /api/games/recommend + /complete
│   │   ├── users.py                ← User auth + profile
│   │   └── rewards.py              ← POST /api/rewards/log — updates bandit
│   │
│   ├── models/                     ← SQLAlchemy database models
│   │   ├── user.py
│   │   ├── session.py
│   │   └── game_log.py
│   │
│   ├── db/                         ← Database connection + migrations
│   │   └── database.py
│   │
│   └── utils/                      ← Shared helpers
│       └── logger.py
│
├── frontend/                       ← React (Vite) web app
│   ├── package.json
│   └── src/
│       ├── components/
│       │   ├── ChatWindow.jsx      ← Chat interface (Week 7)
│       │   ├── GameLauncher.jsx    ← Shows recommended game (Week 7)
│       │   └── MoodDashboard.jsx   ← Stress trend charts (Week 9)
│       │
│       ├── games/                  ← 5 therapeutic mini-games (Weeks 5–6)
│       │   ├── BreathingGame.jsx   ← 4-7-8 breathing pacer
│       │   ├── PuzzleGame.jsx      ← Color nonogram puzzle
│       │   ├── AttentionTap.jsx    ← Mindful tap sequence
│       │   ├── NatureWalk.jsx      ← Guided nature animation
│       │   └── MicroJournal.jsx    ← Expressive writing prompt
│       │
│       ├── pages/
│       │   ├── Home.jsx
│       │   ├── Chat.jsx
│       │   └── Dashboard.jsx
│       │
│       └── hooks/
│           └── useChat.js
│
├── data/
│   ├── raw/                        ← Downloaded datasets (gitignored)
│   │   ├── dreaddit/               ← Stress detection Reddit posts
│   │   ├── go_emotions/            ← Google 27-class emotion labels
│   │   ├── mentalchat16k/          ← Counseling conversation transcripts
│   │   ├── happydb/                ← Positive mood moments
│   │   └── mental_health_reddit/   ← Multi-condition Reddit posts
│   │
│   └── processed/
│       └── synthetic_rl/           ← 2,000 synthetic (state, action, reward) episodes
│
├── notebooks/
│   ├── 01_explore_data.ipynb       ← EDA on all datasets
│   ├── 02_stress_encoder_eval.ipynb← Validate encoder vs Dreaddit labels
│   └── 03_bandit_simulation.ipynb  ← Test bandit on synthetic data
│
└── tests/
    ├── test_stress_encoder.py
    └── test_bandit.py
```

---

## Datasets

| Dataset | Size | Downloaded to | Purpose |
|---------|------|---------------|---------|
| Dreaddit | ~10k posts | `data/raw/dreaddit/` | Stress classifier training & encoder validation |
| GoEmotions | ~58k comments | `data/raw/go_emotions/` | 27-class emotion detection |
| MentalChat16K | ~16k convos | `data/raw/mentalchat16k/` | LLM fine-tuning (chatbot layer) |
| HappyDB | ~100k entries | `data/raw/happydb/` | Positive mood signal for reward calibration |
| Mental Health Reddit | ~7.5k posts | `data/raw/mental_health_reddit/` | Supplementary stress signal |
| Synthetic RL Data | 2,000 episodes | `data/processed/synthetic_rl/` | Bandit pre-training |

Run `python scripts/download_datasets.py --help` for download options.

---

## Tech Stack

**Backend:** Python 3.10 · FastAPI · OpenAI API (GPT-4o-mini) · ChromaDB · SQLite/PostgreSQL

**ML Core:** HuggingFace Transformers · NumPy (LinUCB) · scikit-learn · Datasets

**Frontend:** React (Vite) · Tailwind CSS · Recharts · Phaser.js (games)

**Deploy:** Railway or Render (free tier)

---

## Novel Research Contributions

- **C1 — LLM as RL State Encoder:** GPT-4o-mini extracts a structured stress vector from free-form conversation, feeding a downstream RL agent.
- **C2 — LinUCB for Therapeutic Game Selection:** A contextual bandit learns which therapeutic game works best per stress profile.
- **C3 — Composite Reward Design:** R = α·Δmood + β·engagement + γ·retention − δ·skip_rate

---

## Development Timeline

| Month | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Datasets · Stress encoder · LinUCB bandit · FastAPI | Working ML pipeline |
| 2 | 5 mini-games · React frontend · Full integration | End-to-end web app |
| 3 | Dashboard · Auth · 10-user pilot · Deploy | Live deployed app |
