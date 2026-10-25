# Results (twin-merged benchmark)

Seeds: [0, 1, 2, 3, 4]. Test split (seed 0): 10794 unique requests (3150 attacks, 7644 benign); train 52339, validation 10794. Strict split: False. Values are mean ± s.d. over seeds; all thresholds are chosen on the validation split and applied unchanged to the test split.

Base-calibration chosen by inner CV on training OOF scores: ['isotonic', 'isotonic', 'isotonic', 'isotonic', 'isotonic'].

## Table 1. Detection on the held-out test split

| Method | ROC-AUC | PR-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | Test recall / FPR at val-0.1% thr. | Test recall / FPR at val-1% thr. | ECE |
|---|---|---|---|---|---|---|---|
| Prior pipeline (α=0.6, β=0.4) | 0.9975 ± 0.0002 | 0.9947 ± 0.0007 | 84.25 ± 2.25 | 94.29 ± 1.03 | 84.98 ± 1.88 / 0.102 ± 0.064 | 94.21 ± 1.57 / 1.015 ± 0.254 | 0.302 ± 0.002 |
| Prior pipeline (adaptive fusion) | 0.9968 ± 0.0001 | 0.9934 ± 0.0004 | 83.87 ± 2.56 | 93.74 ± 0.34 | 84.80 ± 2.97 / 0.115 ± 0.074 | 93.66 ± 0.96 / 0.963 ± 0.215 | 0.328 ± 0.002 |
| Prior ML branch (IF/OC-SVM/AE, max) | 0.9690 ± 0.0015 | 0.9345 ± 0.0044 | 31.59 ± 7.70 | 57.50 ± 1.96 | 33.47 ± 3.32 / 0.115 ± 0.046 | 57.71 ± 2.16 / 1.002 ± 0.155 | 0.427 ± 0.003 |
| Prior NLP branch (TF-IDF LR/char SGD/rules, max) | 0.9979 ± 0.0004 | 0.9954 ± 0.0009 | 91.55 ± 0.75 | 96.29 ± 0.34 | 91.45 ± 1.43 / 0.118 ± 0.037 | 96.44 ± 0.28 / 1.094 ± 0.200 | 0.093 ± 0.001 |
| Rules only | 0.7873 ± 0.0024 | 0.7212 ± 0.0035 | 51.59 ± 4.44 | 56.23 ± 0.62 | 51.68 ± 3.99 / 0.055 ± 0.068 | 56.23 ± 0.62 / 0.128 ± 0.048 | 0.157 ± 0.001 |
| CARTA component: char n-gram LR | 0.9987 ± 0.0004 | 0.9972 ± 0.0008 | 93.26 ± 1.81 | 97.14 ± 0.70 | 93.17 ± 1.76 / 0.097 ± 0.033 | 97.18 ± 0.57 / 1.028 ± 0.101 | 0.032 ± 0.001 |
| CARTA component: structure-token LR | 0.9990 ± 0.0002 | 0.9979 ± 0.0004 | 94.05 ± 0.85 | 97.94 ± 0.16 | 94.44 ± 0.83 / 0.131 ± 0.036 | 97.83 ± 0.11 / 0.950 ± 0.102 | 0.022 ± 0.001 |
| CARTA component: GBDT (numeric+rules+profile+context) | 0.9978 ± 0.0010 | 0.9912 ± 0.0053 | 32.06 ± 38.22 | 99.21 ± 0.13 | 31.86 ± 37.96 / 0.037 ± 0.047 | 99.19 ± 0.16 / 0.963 ± 0.148 | 0.005 ± 0.001 |
| CARTA component: Isolation Forest | 0.9563 ± 0.0027 | 0.8973 ± 0.0064 | 14.59 ± 5.92 | 41.73 ± 4.07 | 13.67 ± 7.08 / 0.071 ± 0.066 | 38.09 ± 6.26 / 0.916 ± 0.120 | 0.323 ± 0.003 |
| **CARTA (proposed)** | 0.9996 ± 0.0002 | 0.9994 ± 0.0002 | 98.98 ± 0.45 | 99.64 ± 0.06 | 98.86 ± 0.40 / 0.084 ± 0.050 | 99.62 ± 0.06 / 0.856 ± 0.046 | 0.002 ± 0.000 |

Recall and FPR in %. ECE is computed for every score as if it were a probability (the prior pipeline presents its fused score in [0,1] and thresholds it as such).

Paired bootstrap (1,000 resamples of the test split, per seed), CARTA − prior pipeline:

- seed 0: ΔROC-AUC = +0.0022 (95% CI +0.0017, +0.0029); ΔRecall@FPR1% = +4.03 pp (95% CI +3.12, +5.26)
- seed 1: ΔROC-AUC = +0.0024 (95% CI +0.0018, +0.0030); ΔRecall@FPR1% = +6.06 pp (95% CI +4.75, +7.32)
- seed 2: ΔROC-AUC = +0.0022 (95% CI +0.0017, +0.0027); ΔRecall@FPR1% = +6.47 pp (95% CI +5.14, +7.93)
- seed 3: ΔROC-AUC = +0.0018 (95% CI +0.0014, +0.0023); ΔRecall@FPR1% = +5.13 pp (95% CI +3.29, +7.02)
- seed 4: ΔROC-AUC = +0.0020 (95% CI +0.0015, +0.0026); ΔRecall@FPR1% = +5.45 pp (95% CI +3.92, +7.55)

## Table 2. Prior pipeline at its fixed operating points, 0.6/0.4 fusion with score-only levels (test split)

| Operating point | Recall (%) | FPR (%) | Precision (%) |
|---|---|---|---|
| suspicious_low_plus(final>=0.3) | 100.00 ± 0.00 | 81.54 ± 0.94 | 33.57 ± 0.26 |
| alert_medium_plus(final>=0.5) | 99.91 ± 0.03 | 37.48 ± 0.47 | 52.35 ± 0.31 |
| is_anomaly(final>=0.55) | 99.86 ± 0.05 | 13.27 ± 0.39 | 75.61 ± 0.53 |

Prior pipeline levels (cumulative, level or higher):

| Level | Score-only FPR (%) | Score-only recall (%) | Score×impact FPR (%) | Score×impact recall (%) |
|---|---|---|---|---|
| low | 81.54 ± 0.94 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| medium | 37.48 ± 0.47 | 99.91 ± 0.03 | 93.41 ± 1.21 | 100.00 ± 0.00 |
| high | 10.76 ± 0.25 | 99.83 ± 0.05 | 52.55 ± 0.62 | 99.96 ± 0.01 |
| critical | 1.68 ± 0.18 | 96.62 ± 0.48 | 9.13 ± 0.22 | 31.75 ± 0.83 |

## Table 3. Per-source results (test split)

| Source | n attack / benign (seed 0) | CARTA ROC-AUC | Prior ROC-AUC | CARTA R@FPR1% | Prior R@FPR1% | CARTA R@FPR0.1% | Prior R@FPR0.1% |
|---|---|---|---|---|---|---|---|
| csic2010 | 1250 / 718 | 0.9983 ± 0.0002 | 0.9703 ± 0.0041 | 98.37 ± 0.40 | 71.68 ± 3.70 | 85.60 ± 9.63 | 54.05 ± 5.62 |
| httpparams | 1764 / 2895 | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 | 99.97 ± 0.03 | 99.85 ± 0.11 | 99.94 ± 0.07 | 99.56 ± 0.28 |
| weblogs2019 | 70 / 4031 | 0.9969 ± 0.0058 | 0.9849 ± 0.0080 | 98.84 ± 1.21 | 97.11 ± 1.01 | 94.21 ± 1.05 | 87.59 ± 2.10 |
| webpayloads | 66 / 0 | n/a | n/a | n/a | n/a | n/a | n/a |

Per-source FPR thresholds are computed within the source (threshold-free metric).

Per-family recall at the validation-chosen 1%-FPR threshold:

| Family | n (seed 0) | CARTA (%) | Prior (%) |
|---|---|---|---|
| cmdi | 14 | 94.29 ± 5.98 | 45.71 ± 11.95 |
| cmdinj | 12 | 98.33 ± 3.73 | 53.97 ± 5.56 |
| csic_anomalous | 1250 | 99.39 ± 0.20 | 89.90 ± 3.77 |
| csrf | 14 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| path-traversal | 43 | 100.00 ± 0.00 | 62.79 ± 4.93 |
| rce | 2 | 100.00 ± 0.00 | 60.00 ± 41.83 |
| scanning | 13 | 75.00 ± 8.38 | 14.10 ± 3.63 |
| sqli | 1696 | 99.99 ± 0.03 | 99.36 ± 0.19 |
| ssrf | 12 | 100.00 ± 0.00 | 83.33 ± 10.21 |
| xss | 94 | 100.00 ± 0.00 | 97.25 ± 0.93 |

## Table 4. Conformal risk tiers (calibrated on validation, evaluated on test; cumulative, tier or higher)

| Tier | Nominal budget (%) | CARTA-Mondrian FPR (%) | CARTA-Mondrian recall (%) | CARTA-Mondrian IW-recall (%) | CARTA-global FPR (%) | CARTA-global recall (%) | CARTA-global IW-recall (%) | Prior+conformal (Mondrian) FPR / recall (%) | Prior own levels (0.6/0.4, score-only) FPR / recall (%) |
|---|---|---|---|---|---|---|---|---|---|
| low | 5.00 | 5.113 ± 0.336 | 99.68 ± 0.06 | 99.51 ± 0.10 | 4.976 ± 0.303 | 99.90 ± 0.04 | 99.91 ± 0.04 | 5.330 ± 0.385 / 96.52 ± 1.33 | 81.54 ± 0.94 / 100.00 ± 0.00 |
| medium | 1.00 | 0.950 ± 0.044 | 99.34 ± 0.13 | 98.99 ± 0.21 | 0.856 ± 0.046 | 99.62 ± 0.06 | 99.48 ± 0.11 | 1.138 ± 0.297 / 91.49 ± 0.62 | 37.48 ± 0.47 / 99.91 ± 0.03 |
| high | 0.20 | 0.228 ± 0.039 | 98.02 ± 0.18 | 96.79 ± 0.25 | 0.186 ± 0.041 | 99.25 ± 0.14 | 98.92 ± 0.22 | 0.196 ± 0.092 / 84.68 ± 2.09 | 10.76 ± 0.25 / 99.83 ± 0.05 |
| critical | 0.05 | 0.039 ± 0.021 | 97.71 ± 0.14 | 96.50 ± 0.18 | 0.042 ± 0.023 | 97.81 ± 0.22 | 96.67 ± 0.41 | 0.052 ± 0.052 / 80.33 ± 3.64 | 1.68 ± 0.18 / 96.62 ± 0.48 |

IW-recall = impact-weighted recall, with the impact of each attacked endpoint computed from the prior pipeline's own released impact weights (external to CARTA).

Per-cell view (CARTA, Mondrian): benign FPR against the cell's budget, and recall per cell for Mondrian vs global thresholds:

| Tier | Cell | Pooled into | Cell budget (%) | FPR (%) | Recall, Mondrian (%) | Recall, global (%) |
|---|---|---|---|---|---|---|
| low | high | high | 9.433 ± 0.029 | 8.156 ± 0.969 | 99.06 ± 0.26 | 99.98 ± 0.05 |
| low | low | low | 2.358 ± 0.007 | 0.861 ± 1.192 | 100.00 ± 0.00 | 99.62 ± 0.86 |
| low | std | std | 4.717 ± 0.014 | 5.050 ± 0.340 | 99.92 ± 0.04 | 99.88 ± 0.07 |
| medium | high | high | 1.887 ± 0.006 | 1.653 ± 0.586 | 97.93 ± 0.56 | 99.13 ± 0.28 |
| medium | low | low | 0.472 ± 0.001 | 0.163 ± 0.253 | 100.00 ± 0.00 | 98.80 ± 1.75 |
| medium | std | std | 0.943 ± 0.003 | 0.926 ± 0.069 | 99.89 ± 0.05 | 99.84 ± 0.08 |
| high | high | high | 0.368 ± 0.001 | 0.280 ± 0.131 | 93.45 ± 0.36 | 97.93 ± 0.56 |
| high | low | std | 0.184 ± 0.000 | 0.000 ± 0.000 | 98.80 ± 1.75 | 98.80 ± 1.75 |
| high | std | std | 0.184 ± 0.000 | 0.235 ± 0.043 | 99.85 ± 0.08 | 99.79 ± 0.08 |
| critical | high | std | 0.050 ± 0.000 | 0.280 ± 0.131 | 93.31 ± 0.27 | 93.67 ± 0.81 |
| critical | low | std | 0.050 ± 0.000 | 0.000 ± 0.000 | 97.18 ± 3.26 | 97.18 ± 3.26 |
| critical | std | std | 0.050 ± 0.000 | 0.018 ± 0.020 | 99.49 ± 0.17 | 99.49 ± 0.17 |

Benign validation points per cell (seed 0): {'high': 651, 'low': 384, 'std': 6608}; test: {'high': 641, 'low': 415, 'std': 6588}.

Validity check: held-out data (validation + test) re-split at random into calibration and evaluation halves 200 times per seed, model fixed. Mean benign FPR on the evaluation halves vs the exact split-conformal bound (n+1-k)/(n+1), averaged over seeds:

| Tier | Budget (%) | Mondrian mean FPR (%) | Mondrian exact bound (%) | Global mean FPR (%) | Global exact bound (%) | Mondrian mean recall (%) |
|---|---|---|---|---|---|---|
| low | 5.00 | 4.857 ± 0.035 | 4.980 ± 0.001 | 4.962 ± 0.035 | 4.994 ± 0.000 | 99.68 ± 0.03 |
| medium | 1.00 | 0.946 ± 0.036 | 0.977 ± 0.001 | 0.994 ± 0.010 | 0.994 ± 0.000 | 99.27 ± 0.18 |
| high | 0.20 | 0.181 ± 0.004 | 0.185 ± 0.001 | 0.196 ± 0.006 | 0.196 ± 0.000 | 97.56 ± 0.86 |
| critical | 0.05 | 0.035 ± 0.003 | 0.039 ± 0.000 | 0.040 ± 0.002 | 0.039 ± 0.000 | 95.74 ± 1.68 |

## Table 5. Expected SOC cost per 10,000 events (threshold chosen on validation; cost on test)

| Setting | CARTA | Prior (validation-optimised threshold) | Prior at 0.5 | Prior at 0.55 | CARTA TPR / FPR (%) | Prior-opt TPR / FPR (%) |
|---|---|---|---|---|---|---|
| cfn10_prev1% | 18.8 ± 5.6 | 136.8 ± 7.2 | 3710.9 ± 46.9 | 1315.4 ± 38.1 | 98.90 ± 0.41 / 0.078 ± 0.035 | 90.46 ± 2.39 / 0.419 ± 0.221 |
| cfn100_prev1% | 90.4 ± 12.1 | 467.4 ± 33.4 | 3718.9 ± 47.5 | 1328.0 ± 38.1 | 99.42 ± 0.09 / 0.330 ± 0.124 | 97.66 ± 0.32 / 2.355 ± 0.282 |
| cfn100_prev1%_cap1.5% | 90.4 ± 12.1 | 820.1 ± 125.0 | 3718.9 ± 47.5 | 1328.0 ± 38.1 | 99.42 ± 0.09 / 0.330 ± 0.124 | 92.46 ± 1.35 / 0.665 ± 0.163 |
| cfn20_prev5% | 89.1 ± 11.7 | 456.5 ± 37.5 | 3569.0 ± 45.6 | 1274.9 ± 36.5 | 99.42 ± 0.09 / 0.330 ± 0.124 | 98.11 ± 0.64 / 2.821 ± 0.772 |

Setting cfnX_prevY%: c_FP = 1, c_FN = X, attack prevalence Y; capZ% = alerts limited to Z% of events.

## Table 6. Evasion: attack recall at the clean validation-1%-FPR threshold

| Transform | CARTA (%) | Prior (%) | Requests whose prior-view text changed (%) |
|---|---|---|---|
| clean | 99.62 ± 0.06 | 94.21 ± 1.57 | n/a |
| double_url_encode | 99.87 ± 0.05 | 98.10 ± 1.13 | 91.0 ± 0.4 |
| mixed_case | 99.62 ± 0.06 | 94.45 ± 1.59 | 0.0 ± 0.0 |
| inline_comments | 99.63 ± 0.04 | 96.43 ± 1.33 | 82.1 ± 0.6 |
| fullwidth_unicode | 99.62 ± 0.06 | 94.84 ± 1.62 | 0.0 ± 0.0 |
| whitespace_variants | 99.75 ± 0.04 | 95.49 ± 1.43 | 55.1 ± 0.7 |
| html_entities | 99.62 ± 0.06 | 95.03 ± 1.52 | 0.0 ± 0.0 |
| combined | 99.87 ± 0.05 | 98.15 ± 1.14 | 91.0 ± 0.4 |
| benign_padding | 99.87 ± 0.03 | 92.58 ± 0.47 | 93.8 ± 0.2 |
| adaptive_worst_case | 99.52 ± 0.04 | 89.89 ± 1.58 | n/a |

adaptive_worst_case: per request, the attacker applies whichever transform (or none) minimises the score of the detector under attack (black-box, per-detector).

Fit time (s): prior 65 ± 3, CARTA 655 ± 297.
