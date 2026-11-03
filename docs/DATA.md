# Data
Synthetic only (`src/mmdq/generator.py`), 12,000 rows per seed. Features: amount (lognormal), hour, account_age_days (lognormal), country_mismatch (15%),
merchant_category (8 values), note (3–10 words from 5 topic vocabularies + filler; 30% contain cross-topic words). Label = 1 if
tabular risk + topic risk + N(0, 0.8) − 0.35 > 0. Corruption (`corrupt`): MCAR 20% on amount/hour/account_age; category missing with p=0.3 when account
age < 30 days (MAR); 10% empty notes (missing modality); 15% symmetric label flips in TRAIN only. Injection rates are measured and logged
(actual flip rate 15.2% / 15.6% / 14.7%).
Why synthetic: the candidate public multimodal dataset (UCI Drug Review) was not downloadable at its expected URL on 2026-11-03; see DECISIONS.
Not AndPayments data.
