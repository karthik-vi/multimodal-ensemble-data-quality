"""Imputation, mislabel flagging and multimodal ensembling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .generator import CATS

NUM = ["amount", "hour", "account_age_days"]


# ---------------- imputation ----------------
class Imputer:
    """mean: column mean / most-frequent category.  model: IterativeImputer for numerics (uses all tabular
    columns), and a TF-IDF+tabular classifier for the missing merchant_category (cross-modal: the note text helps)."""

    def __init__(self, kind: str = "mean", seed: int = 0):
        self.kind, self.seed = kind, seed

    def _num(self, df):
        X = df[NUM].copy()
        X["amount"] = np.log(X["amount"])
        X["account_age_days"] = np.log(X["account_age_days"])
        return np.c_[X.values, df["country_mismatch"].values]

    def fit(self, df: pd.DataFrame):
        Xn = self._num(df)
        self.num = (IterativeImputer(max_iter=10, random_state=self.seed) if self.kind == "model" else SimpleImputer()).fit(Xn)
        known = df["merchant_category"].notna()
        self.mode = df.loc[known, "merchant_category"].mode()[0]
        if self.kind == "model":
            self.cat_tfidf = TfidfVectorizer(min_df=2).fit(df["note"])
            Z = self._cat_feats(df[known], self.num.transform(Xn[known.values]))
            self.cat_clf = LogisticRegression(max_iter=2000, C=2.0).fit(Z, df.loc[known, "merchant_category"])
        return self

    def _cat_feats(self, df, num):
        return sparse.hstack([self.cat_tfidf.transform(df["note"]), sparse.csr_matrix(num)]).tocsr()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in NUM:
            out[f"{c}_missing"] = df[c].isna().astype(int)
        out["category_missing"] = df["merchant_category"].isna().astype(int)
        out["text_missing"] = (df["note"].str.len() == 0).astype(int)
        num = self.num.transform(self._num(df))
        out["amount"], out["hour"], out["account_age_days"] = np.exp(num[:, 0]), num[:, 1], np.exp(num[:, 2])
        miss = df["merchant_category"].isna().values
        if miss.any():
            if self.kind == "model":
                out.loc[miss, "merchant_category"] = self.cat_clf.predict(self._cat_feats(df[miss], num[miss]))
            else:
                out.loc[miss, "merchant_category"] = self.mode
        return out


# ---------------- per-modality models ----------------
def tab_matrix(df: pd.DataFrame) -> np.ndarray:
    cat = np.stack([(df["merchant_category"] == c).astype(float).values for c in CATS], 1)
    h = df["hour"].values
    return np.c_[np.log(df["amount"]), h, ((h < 6) | (h > 22)).astype(float), np.log(df["account_age_days"]), df["country_mismatch"],
                 cat, df[[c for c in df.columns if c.endswith("_missing")]].values]


class TabModel:
    def fit(self, df, y, seed=0):
        from xgboost import XGBClassifier

        self.m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                               random_state=seed, n_jobs=2, eval_metric="logloss").fit(tab_matrix(df), y)
        return self

    def proba(self, df):
        return self.m.predict_proba(tab_matrix(df))[:, 1]


class TextTfidf:
    def fit(self, df, y, seed=0):
        self.v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(df["note"])
        self.m = LogisticRegression(max_iter=2000, C=2.0).fit(self.v.transform(df["note"]), y)
        return self

    def proba(self, df):
        return self.m.predict_proba(self.v.transform(df["note"]))[:, 1]


class TextSBERT:
    """Logistic regression on Sentence-BERT embeddings (embeddings are precomputed and passed in via df['_emb'])."""

    def fit(self, df, y, seed=0):
        self.m = LogisticRegression(max_iter=2000, C=1.0).fit(np.stack(df["_emb"].values), y)
        return self

    def proba(self, df):
        return self.m.predict_proba(np.stack(df["_emb"].values))[:, 1]


class EarlyFusion:
    def fit(self, df, y, seed=0):
        self.v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(df["note"])
        self.mu, self.sd = tab_matrix(df).mean(0), tab_matrix(df).std(0) + 1e-6
        self.m = LogisticRegression(max_iter=3000, C=1.0).fit(self._X(df), y)
        return self

    def _X(self, df):
        return sparse.hstack([self.v.transform(df["note"]), sparse.csr_matrix((tab_matrix(df) - self.mu) / self.sd)]).tocsr()

    def proba(self, df):
        return self.m.predict_proba(self._X(df))[:, 1]


BASE = {"tab_xgb": TabModel, "text_tfidf": TextTfidf, "text_sbert": TextSBERT}


def oof_probs(df, y, names, seed=0, folds=5):
    """Out-of-fold probabilities for each base model (used for stacking and for mislabel flagging)."""
    P = np.zeros((len(df), len(names)))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(df, y):
        for j, n in enumerate(names):
            P[va, j] = BASE[n]().fit(df.iloc[tr], y[tr], seed).proba(df.iloc[va])
    return P


class Stacking:
    """Late fusion: base models' OOF probabilities + missing-modality flags -> logistic-regression meta model."""

    def __init__(self, names=("tab_xgb", "text_tfidf", "text_sbert")):
        self.names = list(names)

    def _meta_X(self, P, df):
        logit = np.log(np.clip(P, 1e-4, 1 - 1e-4) / np.clip(1 - P, 1e-4, 1))
        return np.c_[logit, df["text_missing"].values, logit[:, 1:] * (1 - df["text_missing"].values[:, None])]

    def fit(self, df, y, seed=0, oof=None):
        P = oof if oof is not None else oof_probs(df, y, self.names, seed)
        self.meta = LogisticRegression(max_iter=2000).fit(self._meta_X(P, df), y)
        self.base = [BASE[n]().fit(df, y, seed) for n in self.names]
        return self

    def proba(self, df):
        P = np.stack([b.proba(df) for b in self.base], 1)
        return self.meta.predict_proba(self._meta_X(P, df))[:, 1]


# ---------------- mislabel flagging ----------------
def flag_mislabels(p_pos: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """Confident-learning style rule (Northcutt et al., 2021), binary case: per-class threshold t_c = mean
    self-confidence of examples labelled c; flag an example labelled c when p(c) < t_c and p(other) >= t_other."""
    p = np.c_[1 - p_pos, p_pos]
    t = np.array([p[y_obs == c, c].mean() for c in (0, 1)])
    given = p[np.arange(len(y_obs)), y_obs]
    other = p[np.arange(len(y_obs)), 1 - y_obs]
    return (given < t[y_obs]) & (other >= t[1 - y_obs])
