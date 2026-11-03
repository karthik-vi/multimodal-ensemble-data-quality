import numpy as np

from mmdq.generator import corrupt, generate
from mmdq.pipeline import Imputer, Stacking, flag_mislabels


def test_corruption_rates_match_config():
    df = generate(4000, seed=0)
    c, flip = corrupt(df, np.random.default_rng(0), mcar=0.2, text_missing=0.1, label_noise=0.15)
    assert abs(c["amount"].isna().mean() - 0.2) < 0.03
    assert abs((c["note"] == "").mean() - 0.1) < 0.03
    assert abs(flip.mean() - 0.15) < 0.03 and (c["label_observed"] != c["label"]).sum() == flip.sum()


def test_imputers_fill_everything():
    df = generate(1500, seed=1)
    c, _ = corrupt(df, np.random.default_rng(1))
    for kind in ("mean", "model"):
        out = Imputer(kind).fit(c).transform(c)
        assert out[["amount", "hour", "account_age_days", "merchant_category"]].notna().all().all()


def test_flagging_finds_injected_flips():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.where(y == 1, 0.85, 0.15) + rng.normal(0, 0.05, 2000)
    flip = rng.random(2000) < 0.1
    y_obs = np.where(flip, 1 - y, y)
    f = flag_mislabels(np.clip(p, 0, 1), y_obs)
    assert (f & flip).sum() / flip.sum() > 0.85 and (f & flip).sum() / f.sum() > 0.9


def test_stacking_without_sbert_runs():
    df = generate(1200, seed=2)
    c, _ = corrupt(df, np.random.default_rng(2))
    c = Imputer("mean").fit(c).transform(c)
    y = c["label_observed"].values
    m = Stacking(names=("tab_xgb", "text_tfidf")).fit(c.iloc[:900], y[:900])
    acc = ((m.proba(c.iloc[900:]) >= 0.5) == c["label"].values[900:]).mean()
    assert acc > 0.7
