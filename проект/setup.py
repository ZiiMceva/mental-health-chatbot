#!/usr/bin/env python3
"""
One-command project setup script.
Run this ONCE after cloning the repo.

    python setup.py

Does everything:
  1. Creates all directories
  2. Creates Python virtual environment
  3. Installs all dependencies
  4. Creates .env from .env.example
  5. Downloads all datasets
  6. Runs a quick sanity check
"""

import os, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
BLUE  = "\033[94m"; BOLD   = "\033[1m";  RESET = "\033[0m"

def ok(m):   print(f"{GREEN}  ✓ {m}{RESET}")
def warn(m): print(f"{YELLOW}  ⚠ {m}{RESET}")
def info(m): print(f"{BLUE}  → {m}{RESET}")
def err(m):  print(f"{RED}  ✗ {m}{RESET}"); sys.exit(1)
def hdr(m):  print(f"\n{BOLD}{m}{RESET}")

def run(cmd, cwd=None, check=True):
    result = subprocess.run(cmd, shell=True, cwd=cwd,
                            capture_output=True, text=True)
    if check and result.returncode != 0:
        print(result.stderr)
        err(f"Command failed: {cmd}")
    return result

# ── Step 1: Directories ───────────────────────────────────────────────────────
hdr("Step 1/5 — Creating project directories")
dirs = [
    "backend/api", "backend/core", "backend/models",
    "backend/db",  "backend/utils",
    "frontend/src/components", "frontend/src/games",
    "frontend/src/pages",     "frontend/src/hooks",
    "data/raw",    "data/processed",
    "notebooks",   "scripts",    "tests",
]
for d in dirs:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
ok("All directories created")

# ── Step 2: Python virtual environment ───────────────────────────────────────
hdr("Step 2/5 — Python virtual environment")
venv_path = ROOT / "backend" / "venv"
if venv_path.exists():
    warn("venv already exists — skipping creation")
else:
    info("Creating venv...")
    run(f"{sys.executable} -m venv venv", cwd=ROOT / "backend")
    ok("venv created at backend/venv/")

python = str(venv_path / ("Scripts/python" if os.name == "nt" else "bin/python"))
pip    = str(venv_path / ("Scripts/pip"    if os.name == "nt" else "bin/pip"))

info("Installing Python dependencies (this takes 2–3 minutes)...")
run(f'"{pip}" install --upgrade pip --quiet')
run(f'"{pip}" install -r requirements.txt --quiet', cwd=ROOT / "backend")
ok("All Python packages installed")

# ── Step 3: .env file ─────────────────────────────────────────────────────────
hdr("Step 3/5 — Environment file")
env_file     = ROOT / "backend" / ".env"
env_example  = ROOT / "backend" / ".env.example"
if env_file.exists():
    warn(".env already exists — skipping")
else:
    shutil.copy(env_example, env_file)
    ok(".env created from .env.example")
    warn("IMPORTANT: Open backend/.env and add your OPENAI_API_KEY before running the app")

# ── Step 4: .gitignore ────────────────────────────────────────────────────────
hdr("Step 4/5 — .gitignore")
gitignore = ROOT / ".gitignore"
if not gitignore.exists():
    gitignore.write_text(
        "# Python\nvenv/\n__pycache__/\n*.pyc\n.pytest_cache/\n\n"
        "# Env\n.env\n\n# Data (too large for git)\ndata/raw/\ndata/processed/\n\n"
        "# Node\nnode_modules/\ndist/\n\n# IDE\n.vscode/settings.json\n.DS_Store\n"
        "\n# Model weights\n*.npz\n*.pt\n*.bin\n"
    )
    ok(".gitignore created")
else:
    warn(".gitignore already exists — skipping")

# ── Step 5: Download datasets ─────────────────────────────────────────────────
hdr("Step 5/5 — Downloading datasets")
api_key = env_file.read_text()
if "sk-your-key-here" in api_key or "OPENAI_API_KEY=\n" in api_key:
    warn("OPENAI_API_KEY not set — downloading datasets but encoder won't run yet")

downloader = ROOT / "scripts" / "download_datasets.py"
if downloader.exists():
    info("Starting dataset download (5–10 minutes depending on connection)...")
    result = run(f'"{python}" scripts/download_datasets.py', cwd=ROOT, check=False)
    if result.returncode != 0:
        warn("Dataset download had some errors. Check output above.")
    else:
        ok("Datasets downloaded")
else:
    warn("download_datasets.py not found — skipping")

# ── Final instructions ────────────────────────────────────────────────────────
print(f"""
{BOLD}{'='*55}{RESET}
{GREEN}{BOLD}  Setup complete!{RESET}
{BOLD}{'='*55}{RESET}

  Next steps:

  1. Add your OpenAI key:
     {YELLOW}code backend/.env{RESET}   (set OPENAI_API_KEY)

  2. Start the backend:
     {BLUE}cd backend && source venv/bin/activate{RESET}
     {BLUE}uvicorn main:app --reload{RESET}
     → API running at http://localhost:8000
     → Docs at    http://localhost:8000/docs

  3. Start the frontend (in a new terminal):
     {BLUE}cd frontend && npm install && npm run dev{RESET}
     → App running at http://localhost:5173

  4. Test the stress encoder:
     {BLUE}python backend/core/stress_encoder.py{RESET}

  5. Test the bandit:
     {BLUE}python backend/core/bandit.py{RESET}

  Or open VS Code workspace:
     {BLUE}code mental-health-chatbot.code-workspace{RESET}
""")
