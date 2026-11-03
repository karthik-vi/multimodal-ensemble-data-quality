# Experiment log — 2026-11-03, seeds 0–2, CPU (≈80 s per seed). Raw: `results/experiments.json`, `results/run_log.txt`.

## EXP-001 — Baselines vs ensemble under 20% MCAR, 10% missing text, 15% label noise
Accuracy (mean): B1 0.688, B2 0.660, B2b 0.708, B3 0.784, E1 0.785, E2 0.783, E3 0.788, O 0.799. See README table.
- Combining modalities is the big win (+7.6 pts over best single modality). Stacking vs early fusion: +0.1 (noise level).
- Mislabel flagging (confident-learning rule on OOF stacked probabilities): flagged 1,419 / 1,550 / 1,451; true flips 1,215 / 1,247 / 1,177;
  precision 0.52 / 0.49 / 0.49; recall 0.60 / 0.61 / 0.60. Dropping flagged rows: +0.4 pts; closes ~25% of the gap to the clean-label ceiling.
- **Failure:** model-based imputation = mean imputation (amount MAE 74.5 vs 74.5; category accuracy 0.11 / 0.12 / 0.10, chance 0.125). Root cause: independent
  features in the generator. Kept as a documented negative result rather than tuning the generator to make the method look good.
