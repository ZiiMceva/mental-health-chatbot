"""
prepare_dataset.py
==================
Prepares MentalChat16K dataset for LLaMA 3.2 fine-tuning.
Converts 3-column format (instruction, input, output) into
LLaMA chat template format.

Run:
    python scripts/prepare_dataset.py
"""

import pandas as pd
import json
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

GREEN  = "\033[92m"; YELLOW = "\033[93m"
RED    = "\033[91m"; BLUE   = "\033[94m"; RESET = "\033[0m"

def ok(m):   print(f"{GREEN}  ✓ {m}{RESET}")
def warn(m): print(f"{YELLOW}  ⚠ {m}{RESET}")
def info(m): print(f"{BLUE}  → {m}{RESET}")


# ── LLaMA chat template ───────────────────────────────────────────────────────
def format_llama(instruction: str, user_input: str, output: str) -> str:
    """
    Converts 3 fields into LLaMA 3 chat format.
    This is the exact format LLaMA 3.2 expects during fine-tuning.
    """
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{instruction.strip()}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output.strip()}<|eot_id|>"
    )


# ── Load and clean ────────────────────────────────────────────────────────────
def prepare_mentalchat():
    info("Loading MentalChat16K...")
    df = pd.read_csv(DATA_RAW / "mentalchat16k" / "train.csv")
    info(f"Loaded {len(df):,} rows — columns: {df.columns.tolist()}")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["instruction", "input", "output"])
    df = df[df["input"].str.strip().str.len() > 10]
    df = df[df["output"].str.strip().str.len() > 10]
    after = len(df)
    ok(f"After cleaning: {after:,} rows (removed {before - after} empty/short rows)")

    # Format into LLaMA template
    info("Formatting into LLaMA 3.2 chat template...")
    df["text"] = df.apply(
        lambda row: format_llama(row["instruction"], row["input"], row["output"]),
        axis=1
    )

    # Check average length
    avg_len = df["text"].str.len().mean()
    max_len = df["text"].str.len().max()
    info(f"Average text length: {avg_len:.0f} chars | Max: {max_len:,} chars")

    # Filter out very long sequences (>2048 tokens ≈ 8000 chars)
    before = len(df)
    df = df[df["text"].str.len() <= 8000]
    after = len(df)
    if before != after:
        warn(f"Removed {before - after} rows that were too long (>8000 chars)")

    # Train / validation split (90/10)
    val_size  = int(len(df) * 0.1)
    train_df  = df.iloc[:-val_size]
    val_df    = df.iloc[-val_size:]
    ok(f"Train: {len(train_df):,} rows | Validation: {len(val_df):,} rows")

    # Save as JSON (best format for Colab training)
    save_dir = DATA_PROC / "llama_finetune"
    save_dir.mkdir(exist_ok=True)

    # Save full text format (for TRL / SFTTrainer)
    train_records = [{"text": t} for t in train_df["text"].tolist()]
    val_records   = [{"text": t} for t in val_df["text"].tolist()]

    train_path = save_dir / "train.json"
    val_path   = save_dir / "val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    ok(f"Saved train.json   → {train_path}")
    ok(f"Saved val.json     → {val_path}")

    # Also save as CSV (backup)
    train_df[["text"]].to_csv(save_dir / "train.csv", index=False)
    val_df[["text"]].to_csv(save_dir / "val.csv", index=False)
    ok(f"Saved train.csv + val.csv as backup")

    # Print 1 example so you can verify it looks right
    print(f"\n{'='*60}")
    print("  EXAMPLE (first training record):")
    print(f"{'='*60}")
    print(train_df["text"].iloc[0][:800])
    print("  ... [truncated]")
    print(f"{'='*60}\n")

    return save_dir


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(save_dir):
    print(f"\n{'='*60}")
    print("  Dataset Ready for Fine-tuning!")
    print(f"{'='*60}")

    for f in sorted(save_dir.glob("*")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<20} {size_kb:>6} KB")

    print(f"\n  Next step: Upload these files to Google Colab")
    print(f"  Path: {save_dir}")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  LLaMA 3.2 Fine-tune Dataset Preparation")
    print(f"{'='*60}\n")

    save_dir = prepare_mentalchat()
    print_summary(save_dir)
