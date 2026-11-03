# multimodal-ensemble-data-quality

Multimodal (tabular + free text) classification with injected missing values and label noise: imputation, confident-learning-style
mislabel flagging, and a late-fusion stacking ensemble, compared against single-modality and early-fusion baselines.

## What this is / is not

| | |
|---|---|
| **Reconstructs** | Resume (AndPayments, Data Science Intern, Sep – Dec 2023), verbatim: *"Improved model performance from 76.2% to 80.3% via ensemble learning on multimodal data, after developing ML models to impute missing values and flag mislabeled samples."* |
| **Is** | A reconstruction on a **synthetic** tabular+text dataset (`src/mmdq/generator.py`) with documented corruption rates. All metrics measured here. |
| **Is not** | AndPayments' data or model. **Flag:** the resume places this job in the context of *"data quality and searchability for issue-management workflows"* and does not say what the multimodal data was. The payment-risk framing here is my assumption, not a fact from the resume — tell me if it should be re-framed as issue records. |

### Period freeze
AndPayments, **cutoff Dec 2023**. Python 3.11; scikit-learn 1.3.2 (2023-10-23); XGBoost 2.0.2 (2023-11-13); pandas 2.1.4 (2023-12-08); NumPy 1.26.2;
sentence-transformers 2.2.2 + `all-MiniLM-L6-v2@7dbbc903` (2022-11-07 commit); torch 2.1.1. Lock compiled with `--exclude-newer 2023-12-31`.

## Architecture
```
generator (12k rows: amount, hour, account age, country mismatch, merchant category, free-text note; label from tabular + text latent risk)
  └─ corrupt: 20% MCAR on 3 numeric cols, MAR category (30% when account < 30 days), 10% empty notes, 15% symmetric label flips (TRAIN only)
       ├─ impute: mean | model (IterativeImputer + text→category classifier), with missing-indicator features
       ├─ base models: tabular XGBoost | TF-IDF + LR | SBERT + LR   (out-of-fold probabilities)
       ├─ stacking meta-LR on base logits + missing-modality flags
       └─ flag mislabels: per-class confidence thresholds on OOF ensemble probabilities → drop flagged → retrain
  test labels are CLEAN (no injected noise); 8,000 train / 4,000 test; seeds 0–2
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pytest -q
python scripts/run_experiments.py --seeds 0,1,2      # ~4 min CPU
```

## Results (test accuracy on clean labels, mean ± std over 3 seeds)

| Variant | Accuracy | AUC |
|---|---|---|
| B1 tabular XGBoost (mean imputation) | 0.688 ± 0.009 | 0.758 |
| B2 text TF-IDF + LR | 0.660 ± 0.007 | 0.711 |
| B2b text SBERT + LR (best single modality) | 0.708 ± 0.002 | 0.760 |
| B3 early fusion LR | 0.784 ± 0.005 | 0.869 |
| E1 stacking ensemble | 0.785 ± 0.004 | 0.872 |
| E2 + model-based imputation | 0.783 ± 0.005 | 0.871 |
| **E3 + mislabel flagging (drop flagged)** | **0.788 ± 0.004** | **0.874** |
| O reference: same ensemble trained on *clean* labels | 0.799 ± 0.005 | 0.885 |

> Resume claim: 76.2% → 80.3% | Reproduced here on the synthetic dataset (3 seeds): best single modality 70.8% → multimodal ensemble with mislabel flagging 78.8% (clean-label ceiling 79.9%) | Gap / why: different, synthetic data; the gain here comes from combining modalities (+7.6 pts, early fusion vs best single modality); imputation added nothing and flagging +0.4 pts.

Mislabel flagging finds 60% of the injected flips at ~50% precision (EXPERIMENTS).

## Limitations / failure cases
- **Imputation could not help by construction:** the generator samples tabular features independently, so there is no signal to impute from
  (model ≈ mean imputation; category imputation accuracy ≈ 11% vs 12.5% chance). A realistic generator needs correlated features.
- Early fusion is nearly as good as stacking here; the "ensemble" gain is mostly a "multimodal" gain.
- Synthetic data; label noise is symmetric and uniform (real noise is class- and annotator-dependent).

## How to extend
Correlated generator or a real multimodal dataset; relabel-instead-of-drop; class-conditional noise; per-modality dropout robustness tests.
