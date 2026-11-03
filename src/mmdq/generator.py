"""Synthetic multimodal (tabular + free-text) payment-risk dataset with controlled corruption.

NOT AndPayments data. Generative story: each transaction has a latent risk score built from tabular signals
(amount, hour, account age, country mismatch, merchant category) AND from a text signal (the free-text
customer/merchant note, whose words depend on a hidden "intent" topic). The label is 1 (risky) if the latent
score + noise > 0. Neither modality alone recovers the label; together they do better -> ensembles help.

Corruptions (training split only for labels; features corrupted in all splits):
  MCAR missingness   : `mcar` fraction of account_age_days, amount, hour set to NaN independently
  MAR missingness    : merchant_category missing with prob 0.3 when account_age_days < 30 (new merchants)
  missing modality   : `text_missing` fraction of notes replaced by ""
  label noise        : `label_noise` fraction of TRAIN labels flipped (symmetric); the flip mask is returned
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATS = ["grocery", "electronics", "travel", "gaming", "jewelry", "utilities", "fashion", "crypto_exchange"]
CAT_RISK = {"grocery": -0.6, "electronics": 0.3, "travel": 0.2, "gaming": 0.5, "jewelry": 0.6, "utilities": -0.8, "fashion": 0.0, "crypto_exchange": 0.9}
TOPICS = {  # word pools; risky topics push the latent score up
    "routine": (-1.0, "monthly bill payment regular order groceries subscription renewal invoice usual weekly household".split()),
    "gift": (0.2, "gift birthday present friend surprise voucher card wrap send family".split()),
    "urgent": (0.9, "urgent asap immediately quick fast now rush priority today hurry".split()),
    "resale": (1.1, "bulk resale multiple units wholesale reseller stock quantity lot flip".split()),
    "account": (1.3, "new account changed address shipping different verify card added recently".split()),
}
FILLER = "please thanks order item payment the for my to and a of".split()


def generate(n: int = 12000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = np.exp(rng.normal(4.0, 1.1, n))
    hour = rng.integers(0, 24, n)
    age = np.exp(rng.normal(5.0, 1.4, n)).clip(1, 3650)
    mismatch = (rng.random(n) < 0.15).astype(int)
    cat = rng.choice(CATS, n)
    topic_names = list(TOPICS)
    topic = rng.choice(topic_names, n, p=[0.45, 0.2, 0.15, 0.1, 0.1])
    notes = []
    for t in topic:
        words = list(rng.choice(TOPICS[t][1], rng.integers(3, 7))) + list(rng.choice(FILLER, rng.integers(2, 6)))
        if rng.random() < 0.3:  # cross-topic words make the text signal imperfect
            words += list(rng.choice(TOPICS[rng.choice(topic_names)][1], 2))
        rng.shuffle(words)
        notes.append(" ".join(words))
    tab = (0.55 * (np.log(amount) - 4) + 0.5 * ((hour < 6) | (hour > 22)) - 0.45 * (np.log(age) - 5)
           + 0.9 * mismatch + np.vectorize(CAT_RISK.get)(cat))
    txt = np.vectorize(lambda t: TOPICS[t][0])(topic)
    latent = tab + txt + rng.normal(0, 0.8, n) - 0.35
    return pd.DataFrame({"amount": amount.round(2), "hour": hour.astype(float), "account_age_days": age.round(0),
                         "country_mismatch": mismatch, "merchant_category": cat, "note": notes,
                         "label": (latent > 0).astype(int), "topic": topic})


def corrupt(df: pd.DataFrame, rng: np.random.Generator, mcar: float = 0.2, text_missing: float = 0.1,
            label_noise: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    for c in ("account_age_days", "amount", "hour"):
        df.loc[rng.random(len(df)) < mcar, c] = np.nan
    new = df["account_age_days"].fillna(9999) < 30
    df["merchant_category"] = df["merchant_category"].astype(object)
    df.loc[new & (rng.random(len(df)) < 0.3), "merchant_category"] = None
    df.loc[rng.random(len(df)) < text_missing, "note"] = ""
    flip = rng.random(len(df)) < label_noise
    df["label_observed"] = np.where(flip, 1 - df["label"], df["label"])
    return df, flip
