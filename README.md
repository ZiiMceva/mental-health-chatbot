# Stress-aware recommendation of supportive activities

Code accompanying the manuscript *Adaptive stress-aware recommendation of supportive engagement activities for digital mental health: a contextual bandit approach with domain-specific language models* (Makaeva Ziinat, Muhammad Ahmed, Md Ariful Islam Mozumder, Sabeeha Jabeen, Hee-Cheol Kim; Inje University).

The pipeline detects stress and emotion in free text with domain-adapted language models (MentalRoBERTa) and selects a short supportive activity with a contextual bandit (LinUCB). The repository contains the training and evaluation notebooks, the engagement simulator and bandit experiments, and a prototype web application.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/multi_seed_validation.ipynb` | Five-seed encoder comparison (BERT-base, RoBERTa-base, MentalRoBERTa) for stress and emotion, taxonomy ablation, calibration and temperature scaling, error propagation, stress-threshold sensitivity |
| `notebooks/mental_roberta_v3.ipynb` | MentalRoBERTa training on Dreaddit, K-means and zero-shot LLM alternatives, LLM-as-judge evaluation |
| `notebooks/mental_roberta_v4_goemotion.ipynb` | GoEmotions-to-five-class mapping and emotion classifier |
| `notebooks/reward_v2_experiment.ipynb` | Engagement simulator (reward v2) and population-level bandit comparison |
| `notebooks/03_bandit_personalisation_experiment.ipynb` | Shared vs. per-user LinUCB, corrected configuration, sensitivity analysis |
| `notebooks/real_user_validation_clean.ipynb` | External validation against psychologist ratings and DASS-21 (outputs removed; participant data are not public) |
| `notebooks/llm-Chat.ipynb`, `notebooks/mental_roberta_encoder.ipynb` | Earlier development notebooks |
| `notebooks/*_results.json`, `data/processed/*.json` | Saved results per seed |
| `backend/` | FastAPI prototype (stress encoder, bandit, session memory) |
| `frontend/` | React prototype with the activity games |
| `scripts/` | Dataset download and preparation |
| `tests/` | Unit tests for the bandit and stress encoder |

## Data

- Dreaddit (Turcan & McKeown, 2019) and GoEmotions (Demszky et al., 2020) are public; `scripts/download_datasets.py` downloads them into `data/raw/`.
- The external validation survey contains free-text self-reports and is not included. De-identified data are available from the corresponding author on reasonable request, subject to institutional approval.
- Trained model weights are not included because of their size.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on Linux/macOS
pip install -r requirements-notebooks.txt
python scripts/download_datasets.py
```

Experiments in the manuscript were run with Python 3.12.1, PyTorch 2.6.0 (CUDA 12.4), Transformers 5.8.0 and scikit-learn 1.8.0 on one NVIDIA GeForce GTX 1070 GPU. Seeds: 42, 123, 456, 789, 2024 (classifiers); the bandit experiments use 5 or 20 seeds as stated in each notebook.

API keys (Hugging Face, Groq) are read from placeholders `YOUR_HF_TOKEN` and `YOUR_GROQ_API_KEY`; set your own before running the LLM cells.

## Citation

If you use this code, please cite the manuscript (citation details will be added after publication) and the archived release (see `CITATION.cff`).

## Licence

MIT, see `LICENSE`.
