"""
Dataset Auto-Downloader
========================
Run this once at project start. Downloads all datasets needed for the project.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets dreaddit go_emotions   # specific only
    python scripts/download_datasets.py --skip-heavy                       # skip large datasets
"""

import os
import json
import argparse
import time
from pathlib import Path

# ── Colour helpers for terminal output ───────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  ✓ {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  ⚠ {msg}{RESET}")
def err(msg):   print(f"{RED}  ✗ {msg}{RESET}")
def info(msg):  print(f"{BLUE}  → {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}")


# ── Auto-install missing packages ─────────────────────────────────────────────
def ensure_packages():
    import importlib, subprocess, sys
    required = {
        "datasets":   "datasets",
        "huggingface_hub": "huggingface_hub",
        "pandas":     "pandas",
        "requests":   "requests",
        "tqdm":       "tqdm",
    }
    for import_name, pip_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            info(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   pip_name, "--quiet"])
            ok(f"{pip_name} installed")

ensure_packages()

# ── Now safe to import ────────────────────────────────────────────────────────
import pandas as pd
import requests
from tqdm import tqdm
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET DOWNLOADERS
# ══════════════════════════════════════════════════════════════════════════════

def download_dreaddit():
    """
    Dreaddit — Reddit stress detection dataset.
    10,040 posts from 5 stress-related subreddits, binary labeled (stress/no-stress).
    HuggingFace: https://huggingface.co/datasets/dreaddit
    """
    header("1/5  Dreaddit — Stress Detection (Reddit)")
    save_dir = DATA_RAW / "dreaddit"
    save_dir.mkdir(exist_ok=True)

    if (save_dir / "train.csv").exists():
        warn("Already downloaded — skipping."); return

    info("Loading from HuggingFace datasets hub...")
    try:
        ds = load_dataset("andreagasparini/dreaddit")

        for split in ds.keys():
            df = ds[split].to_pandas()
            out = save_dir / f"{split}.csv"
            df.to_csv(out, index=False)
            ok(f"Saved {split}.csv  ({len(df):,} rows)")

        # Save metadata
        meta = {
            "name": "Dreaddit",
            "splits": {s: len(ds[s]) for s in ds.keys()},
            "columns": ds["train"].column_names,
            "label_col": "label",
            "text_col": "text",
            "labels": {"0": "no stress", "1": "stress"},
            "use": "Primary training data for stress classifier / LLM encoder validation",
            "source": "https://huggingface.co/datasets/dreaddit",
        }
        (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        ok("Dreaddit complete!\n")

    except Exception as e:
        err(f"Dreaddit failed: {e}")
        warn("Manual download: https://huggingface.co/datasets/dreaddit")


def download_go_emotions():
    """
    GoEmotions — Google fine-grained emotion dataset.
    58,009 Reddit comments labeled with 27 emotion categories.
    HuggingFace: https://huggingface.co/datasets/go_emotions
    """
    header("2/5  GoEmotions — Emotion Classification (Google)")
    save_dir = DATA_RAW / "go_emotions"
    save_dir.mkdir(exist_ok=True)

    if (save_dir / "train.csv").exists():
        warn("Already downloaded — skipping."); return

    info("Loading from HuggingFace datasets hub...")
    try:
        # 'simplified' config: 28 emotions collapsed to 28 binary columns
        ds = load_dataset("go_emotions", "simplified")

        emotions = ds["train"].features["labels"].feature.names

        for split in ds.keys():
            df = ds[split].to_pandas()
            # Expand multi-label list into readable string
            df["emotion_names"] = df["labels"].apply(
                lambda ids: [emotions[i] for i in ids]
            )
            out = save_dir / f"{split}.csv"
            df.to_csv(out, index=False)
            ok(f"Saved {split}.csv  ({len(df):,} rows)")

        meta = {
            "name": "GoEmotions",
            "splits": {s: len(ds[s]) for s in ds.keys()},
            "emotion_classes": emotions,
            "text_col": "text",
            "label_col": "labels",
            "use": "Fine-grained emotion classification for stress state vector emotion_type field",
            "source": "https://huggingface.co/datasets/go_emotions",
        }
        (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        ok("GoEmotions complete!\n")

    except Exception as e:
        err(f"GoEmotions failed: {e}")
        warn("Manual: https://huggingface.co/datasets/go_emotions")


def download_mentalchat16k():
    """
    MentalChat16K — Mental health counseling conversations.
    Real anonymized coach-caregiver transcripts + GPT-3.5 synthetic counseling dialogues.
    HuggingFace: https://huggingface.co/datasets/ShenLab/MentalChat16K
    """
    header("3/5  MentalChat16K — Counseling Conversations")
    save_dir = DATA_RAW / "mentalchat16k"
    save_dir.mkdir(exist_ok=True)

    if (save_dir / "train.csv").exists():
        warn("Already downloaded — skipping."); return

    info("Loading from HuggingFace datasets hub...")
    try:
        ds = load_dataset("ShenLab/MentalChat16K")

        for split in ds.keys():
            df = ds[split].to_pandas()
            out = save_dir / f"{split}.csv"
            df.to_csv(out, index=False)
            ok(f"Saved {split}.csv  ({len(df):,} rows)")

        meta = {
            "name": "MentalChat16K",
            "splits": {s: len(ds[s]) for s in ds.keys()},
            "columns": ds["train"].column_names,
            "use": "LLM chatbot layer fine-tuning for empathetic counseling responses",
            "source": "https://huggingface.co/datasets/ShenLab/MentalChat16K",
        }
        (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        ok("MentalChat16K complete!\n")

    except Exception as e:
        err(f"MentalChat16K failed: {e}")
        warn("Manual: https://huggingface.co/datasets/ShenLab/MentalChat16K")


def download_happydb():
    """
    HappyDB — Crowd-sourced happiness moments.
    100,535 entries of what made people happy in the last 24h / 3 months.
    GitHub: https://github.com/megagonlabs/HappyDB
    """
    header("4/5  HappyDB — Positive Mood Moments")
    save_dir = DATA_RAW / "happydb"
    save_dir.mkdir(exist_ok=True)

    if (save_dir / "cleaned_hm.csv").exists():
        warn("Already downloaded — skipping."); return

    files = {
        "cleaned_hm.csv": (
            "https://raw.githubusercontent.com/megagonlabs/"
            "HappyDB/master/happydb/data/cleaned_hm.csv"
        ),
        "demographic.csv": (
            "https://raw.githubusercontent.com/megagonlabs/"
            "HappyDB/master/happydb/data/demographic.csv"
        ),
    }

    for fname, url in files.items():
        info(f"Downloading {fname}...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            (save_dir / fname).write_bytes(resp.content)
            df = pd.read_csv(save_dir / fname)
            ok(f"Saved {fname}  ({len(df):,} rows)")
        except Exception as e:
            err(f"Failed {fname}: {e}")

    meta = {
        "name": "HappyDB",
        "files": list(files.keys()),
        "text_col": "cleaned_hm",
        "use": "Positive mood signal for calibrating post-game reward baseline and mood shift detection",
        "source": "https://github.com/megagonlabs/HappyDB",
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    ok("HappyDB complete!\n")


def download_mentalhealth_reddit():
    """
    Mental Health Reddit — Multi-subreddit posts about anxiety, depression, stress.
    Kaggle mirror available on HuggingFace: https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned
    Also loads a general mental health Twitter/Reddit composite set.
    """
    header("5/5  Mental Health Reddit — Multi-condition Posts")
    save_dir = DATA_RAW / "mental_health_reddit"
    save_dir.mkdir(exist_ok=True)

    if (save_dir / "train.csv").exists():
        warn("Already downloaded — skipping."); return

    info("Loading depression Reddit dataset from HuggingFace...")
    try:
        ds = load_dataset("dair-ai/emotion")

        for split in ds.keys():
            df = ds[split].to_pandas()
            out = save_dir / f"{split}.csv"
            df.to_csv(out, index=False)
            ok(f"Saved {split}.csv  ({len(df):,} rows)")

        meta = {
            "name": "Mental Health Reddit",
            "splits": {s: len(ds[s]) for s in ds.keys()},
            "use": "Supplementary stress/depression signal for encoder training and augmentation",
            "source": "https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned",
        }
        (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        ok("Mental Health Reddit complete!\n")

    except Exception as e:
        err(f"Mental Health Reddit failed: {e}")
        warn("Manual: https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned")


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATASET GENERATOR (for RL simulation — no internet needed)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_rl_data():
    """
    Generates synthetic (state, action, reward) tuples for pre-training
    the LinUCB bandit before real users are available.
    Creates 2,000 simulated user-session records.
    """
    header("★  Generating Synthetic RL Training Data")
    import random, numpy as np

    save_dir = DATA_PROC / "synthetic_rl"
    save_dir.mkdir(exist_ok=True)

    if (save_dir / "synthetic_episodes.csv").exists():
        warn("Already generated — skipping."); return

    random.seed(42)
    np.random.seed(42)

    GAMES = ["breathing", "puzzle", "attention_tap", "nature_walk", "journal"]

    # Ground-truth reward table: which game works best for which emotion
    # Rows = emotion_type, Cols = game  (base reward, varies per user)
    REWARD_TABLE = {
        #                  breath  puzzle  tap   nature  journal
        "anxiety":        [0.85,   0.55,   0.65,  0.60,   0.40],
        "burnout":        [0.60,   0.45,   0.50,  0.85,   0.55],
        "sadness":        [0.50,   0.40,   0.45,  0.65,   0.85],
        "overwhelm":      [0.80,   0.70,   0.55,  0.60,   0.50],
        "neutral":        [0.55,   0.65,   0.60,  0.55,   0.60],
    }
    EMOTIONS = list(REWARD_TABLE.keys())

    records = []
    for episode in range(2000):
        # Sample a random user-state
        emotion    = random.choice(EMOTIONS)
        stress_lvl = round(random.uniform(0.3, 0.95), 3)
        cog_load   = round(random.uniform(0.2, 0.9), 3)
        engagement = round(random.uniform(0.4, 1.0), 3)
        session_n  = random.randint(1, 20)

        # Sample action (game chosen)
        action_idx = random.randint(0, 4)
        game       = GAMES[action_idx]

        # Compute reward with noise
        base_reward = REWARD_TABLE[emotion][action_idx]
        # Higher stress → game works better if it's a good match
        stress_bonus = (stress_lvl - 0.5) * 0.1
        noise        = np.random.normal(0, 0.08)
        reward       = float(np.clip(base_reward + stress_bonus + noise, 0, 1))

        records.append({
            "episode":      episode,
            "stress_level": stress_lvl,
            "emotion_type": emotion,
            "cognitive_load": cog_load,
            "engagement":   engagement,
            "session_count": session_n,
            "action":       action_idx,
            "game_name":    game,
            "reward":       round(reward, 4),
            "is_optimal":   action_idx == REWARD_TABLE[emotion].index(max(REWARD_TABLE[emotion])),
        })

    df = pd.DataFrame(records)
    df.to_csv(save_dir / "synthetic_episodes.csv", index=False)

    # Summary stats
    summary = df.groupby(["emotion_type", "game_name"])["reward"].mean().unstack()
    summary.to_csv(save_dir / "reward_summary_by_emotion_game.csv")

    meta = {
        "records": len(df),
        "games": GAMES,
        "emotions": EMOTIONS,
        "reward_table": REWARD_TABLE,
        "use": "Pre-trains the LinUCB bandit agent before real user data is available",
        "optimal_policy": {e: GAMES[REWARD_TABLE[e].index(max(REWARD_TABLE[e]))] for e in EMOTIONS},
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    ok(f"Generated 2,000 synthetic episodes → data/processed/synthetic_rl/\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    header("=" * 55)
    print(f"{BOLD}  Download Summary{RESET}")
    header("=" * 55)

    datasets = {
        "dreaddit":              DATA_RAW  / "dreaddit"         / "train.csv",
        "go_emotions":           DATA_RAW  / "go_emotions"      / "train.csv",
        "mentalchat16k":         DATA_RAW  / "mentalchat16k"    / "train.csv",
        "happydb":               DATA_RAW  / "happydb"          / "cleaned_hm.csv",
        "mental_health_reddit":  DATA_RAW  / "mental_health_reddit" / "train.csv",
        "synthetic_rl_data":     DATA_PROC / "synthetic_rl"     / "synthetic_episodes.csv",
    }

    all_ok = True
    for name, path in datasets.items():
        if path.exists():
            rows = len(pd.read_csv(path))
            ok(f"{name:<28}  {rows:>7,} rows  →  {path.relative_to(ROOT)}")
        else:
            err(f"{name:<28}  MISSING")
            all_ok = False

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}  All datasets ready. Run notebooks/01_explore_data.ipynb next.{RESET}")
    else:
        warn("Some datasets failed. Check errors above and retry or download manually.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

ALL_DOWNLOADERS = {
    "dreaddit":             download_dreaddit,
    "go_emotions":          download_go_emotions,
    "mentalchat16k":        download_mentalchat16k,
    "happydb":              download_happydb,
    "mental_health_reddit": download_mentalhealth_reddit,
    "synthetic":            generate_synthetic_rl_data,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all project datasets")
    parser.add_argument("--datasets", nargs="+", choices=list(ALL_DOWNLOADERS.keys()),
                        help="Download specific datasets only")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip mentalchat16k (largest dataset, ~300MB)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}  Mental Health Chatbot — Dataset Downloader{RESET}")
    print(f"{BOLD}{'='*55}{RESET}")
    print(f"  Saving to: {DATA_RAW}")

    targets = args.datasets or list(ALL_DOWNLOADERS.keys())
    if args.skip_heavy and "mentalchat16k" in targets:
        targets.remove("mentalchat16k")
        warn("Skipping mentalchat16k (--skip-heavy flag)")

    start = time.time()
    for key in targets:
        ALL_DOWNLOADERS[key]()

    print_summary()
    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.1f}s\n")
