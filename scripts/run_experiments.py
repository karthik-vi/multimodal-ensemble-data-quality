"""Baseline vs ensemble experiment with injected missing values and label noise.

For each seed: generate 12,000 rows -> split 8,000 train / 4,000 test -> corrupt features everywhere and labels in
TRAIN only (test is scored against the clean labels) -> run variants:

  B1 tab-only XGBoost, mean imputation            B2 text-only TF-IDF + LR
  B3 early fusion (TF-IDF + tabular) LR, mean imp  E1 stacking ensemble (tab XGB + TF-IDF LR + SBERT LR), mean imp
  E2 E1 + model-based imputation                    E3 E2 + confident-learning mislabel removal
  O  E2 trained on CLEAN labels (reference upper bound, not achievable in practice)

  python scripts/run_experiments.py --seeds 0,1,2 --out results/experiments.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdq.generator import corrupt, generate  # noqa: E402
from mmdq.pipeline import NUM, BASE, EarlyFusion, Imputer, Stacking, flag_mislabels, oof_probs  # noqa: E402

SBERT = ("sentence-transformers/all-MiniLM-L6-v2", "7dbbc90392e2f80f3d3c277d6e90027e55de9125")  # pre-2024 revision


def embed(texts):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(snapshot_download(SBERT[0], revision=SBERT[1], ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.ot", "*.msgpack"]), device="cpu")
    E = m.encode(list(texts), batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    E[np.array([len(t) == 0 for t in texts])] = 0.0  # missing modality -> zero vector
    return E


def score(y, p):
    return {"accuracy": round(accuracy_score(y, p >= 0.5), 4), "f1": round(f1_score(y, p >= 0.5), 4), "auc": round(roc_auc_score(y, p), 4)}


def run_seed(seed, mcar, text_missing, label_noise):
    rng = np.random.default_rng(seed + 1000)
    full = generate(12000, seed=seed)
    clean_train, clean_test = full.iloc[:8000].reset_index(drop=True), full.iloc[8000:].reset_index(drop=True)
    tr, flip = corrupt(clean_train, rng, mcar, text_missing, label_noise)
    te, _ = corrupt(clean_test, rng, mcar, text_missing, 0.0)
    E = embed(list(tr["note"]) + list(te["note"]))
    tr["_emb"], te["_emb"] = list(E[:8000]), list(E[8000:])
    y_obs, y_te = tr["label_observed"].values, te["label"].values
    out = {"label_noise_rate_actual": round(float(flip.mean()), 4)}
    imp = {k: Imputer(k, seed).fit(tr) for k in ("mean", "model")}
    T = {k: (imp[k].transform(tr), imp[k].transform(te)) for k in imp}
    # imputation quality on the cells that were actually removed
    iq = {}
    for k in imp:
        d = {}
        for c in NUM:
            m = tr[c].isna().values
            truth, est = clean_train.loc[m, c].values, T[k][0].loc[m, c].values
            d[f"{c}_mae"] = round(float(np.mean(np.abs(truth - est))), 3)
        m = tr["merchant_category"].isna().values
        d["merchant_category_acc"] = round(float((T[k][0].loc[m, "merchant_category"].values == clean_train.loc[m, "merchant_category"].values).mean()), 4)
        iq[k] = d
    out["imputation_quality_train"] = iq

    trm, tem = T["mean"]
    out["B1_tab_xgb_mean"] = score(y_te, BASE["tab_xgb"]().fit(trm, y_obs, seed).proba(tem))
    out["B2_text_tfidf"] = score(y_te, BASE["text_tfidf"]().fit(trm, y_obs, seed).proba(tem))
    out["B2b_text_sbert"] = score(y_te, BASE["text_sbert"]().fit(trm, y_obs, seed).proba(tem))
    out["B3_early_fusion_mean"] = score(y_te, EarlyFusion().fit(trm, y_obs, seed).proba(tem))
    out["E1_stack_mean"] = score(y_te, Stacking().fit(trm, y_obs, seed).proba(tem))
    trd, ted = T["model"]
    oof = oof_probs(trd, y_obs, Stacking().names, seed)
    st = Stacking().fit(trd, y_obs, seed, oof=oof)
    out["E2_stack_model_impute"] = score(y_te, st.proba(ted))
    # mislabel flagging with OOF *stacked* probabilities (meta model fitted OOF as well)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    p_oof = cross_val_predict(LogisticRegression(max_iter=2000), st._meta_X(oof, trd), y_obs, cv=5, method="predict_proba")[:, 1]
    flagged = flag_mislabels(p_oof, y_obs)
    tp = int((flagged & flip).sum())
    out["mislabel_flagging"] = {"n_flagged": int(flagged.sum()), "n_true_flips": int(flip.sum()),
                                "precision": round(tp / max(1, flagged.sum()), 4), "recall": round(tp / max(1, flip.sum()), 4)}
    keep = ~flagged
    out["E3_stack_model_impute_clean_flagged"] = score(y_te, Stacking().fit(trd[keep].reset_index(drop=True), y_obs[keep], seed).proba(ted))
    out["O_reference_clean_labels"] = score(y_te, Stacking().fit(trd, clean_train["label"].values, seed).proba(ted))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--mcar", type=float, default=0.2)
    ap.add_argument("--text-missing", type=float, default=0.1)
    ap.add_argument("--label-noise", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("results/experiments.json"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = {}
    for s in seeds:
        t0 = time.time()
        runs[s] = run_seed(s, a.mcar, a.text_missing, a.label_noise)
        print(s, {k: v.get("accuracy", v) for k, v in runs[s].items() if isinstance(v, dict) and "accuracy" in v}, f"{time.time() - t0:.0f}s", flush=True)
    summary = {}
    for k in runs[seeds[0]]:
        if isinstance(runs[seeds[0]][k], dict) and "accuracy" in runs[seeds[0]][k]:
            summary[k] = {m: {"mean": round(float(np.mean([runs[s][k][m] for s in seeds])), 4), "std": round(float(np.std([runs[s][k][m] for s in seeds])), 4)}
                          for m in ("accuracy", "f1", "auc")}
    rep = {"config": vars(a) | {"out": str(a.out)}, "summary": summary, "runs": runs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2))
    for k, v in summary.items():
        print(f"{k:38s} acc {v['accuracy']['mean']:.4f}±{v['accuracy']['std']:.4f}  auc {v['auc']['mean']:.4f}")


if __name__ == "__main__":
    main()
