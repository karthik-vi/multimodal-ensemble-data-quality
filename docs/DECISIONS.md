# Decisions
**Period freeze:** AndPayments, cutoff Dec 2023 — see README; all versions checked on PyPI (latest 2023-12-08); SBERT model pinned to a 2022-11-07 Hub commit.

- **ADR-001 Synthetic multimodal data (2026-11-03).** Tried UCI Drug Review (text + tabular, CC BY 4.0): download URL returned 404, API lookup failed.
  Fallback: documented generator with exact control over missingness and label noise. Domain framing (payment risk) is an assumption — the resume
  only says "multimodal data" in an issue-management context. Would change: user guidance or a downloadable real dataset.
- **ADR-002 XGBoost + TF-IDF/SBERT logistic models, late-fusion stacking (2026-11-03).** Standard 2023 tools; stacking handles a missing modality via flags.
- **ADR-003 Confident-learning-style flagging (Northcutt et al., 2021) implemented directly (2026-11-03).** Avoids a dependency; binary-case rule.
- **ADR-004 Evaluation on clean test labels, 3 seeds, with a clean-label ceiling (2026-11-03).**
- **ADR-005 Deployment: scripts only (2026-11-03).** Offline data-quality tooling; no service.
