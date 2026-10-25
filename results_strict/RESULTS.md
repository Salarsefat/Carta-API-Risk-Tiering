# Strict split (no background rule), seeds 0-2

Seeds: [0, 1, 2]. Test split (seed 0): 12869 unique requests (4213 attacks, 8656 benign); train 60054, validation 12869. Strict split: True. Values are mean ± s.d. over seeds; all thresholds are chosen on the validation split and applied unchanged to the test split.

Base-calibration chosen by inner CV on training OOF scores: ['isotonic', 'isotonic', 'isotonic'].

## Table 1. Detection on the held-out test split

| Method | ROC-AUC | PR-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | Test recall / FPR at val-0.1% thr. | Test recall / FPR at val-1% thr. | ECE |
|---|---|---|---|---|---|---|---|
| Prior pipeline (α=0.6, β=0.4) | 0.9972 ± 0.0004 | 0.9941 ± 0.0010 | 75.80 ± 0.79 | 94.56 ± 0.74 | 82.77 ± 1.59 / 0.146 ± 0.029 | 94.72 ± 0.17 / 1.078 ± 0.207 | 0.301 ± 0.001 |
| Prior pipeline (adaptive fusion) | 0.9947 ± 0.0013 | 0.9895 ± 0.0028 | 75.91 ± 4.44 | 91.85 ± 1.98 | 78.72 ± 5.65 / 0.158 ± 0.047 | 92.33 ± 1.60 / 1.074 ± 0.064 | 0.322 ± 0.001 |
| Prior ML branch (IF/OC-SVM/AE, max) | 0.9174 ± 0.0044 | 0.8350 ± 0.0065 | 9.89 ± 0.61 | 31.77 ± 0.57 | 8.92 ± 0.84 / 0.065 ± 0.027 | 32.87 ± 0.52 / 1.248 ± 0.151 | 0.382 ± 0.002 |
| Prior NLP branch (TF-IDF LR/char SGD/rules, max) | 0.9981 ± 0.0004 | 0.9959 ± 0.0010 | 75.24 ± 18.38 | 98.16 ± 0.12 | 94.09 ± 0.58 / 0.166 ± 0.007 | 98.24 ± 0.21 / 1.078 ± 0.107 | 0.097 ± 0.001 |
| Rules only | 0.7268 ± 0.0036 | 0.6599 ± 0.0049 | 39.27 ± 0.54 | 44.68 ± 0.69 | 39.27 ± 0.54 / 0.031 ± 0.018 | 44.68 ± 0.69 / 0.150 ± 0.042 | 0.196 ± 0.002 |
| CARTA component: char n-gram LR | 0.9982 ± 0.0004 | 0.9966 ± 0.0005 | 73.77 ± 14.35 | 98.40 ± 0.07 | 82.59 ± 18.58 / 0.123 ± 0.055 | 98.44 ± 0.14 / 1.017 ± 0.174 | 0.042 ± 0.001 |
| CARTA component: structure-token LR | 0.9986 ± 0.0002 | 0.9975 ± 0.0004 | 88.21 ± 1.92 | 98.91 ± 0.14 | 94.90 ± 1.69 / 0.146 ± 0.018 | 98.89 ± 0.14 / 0.951 ± 0.052 | 0.034 ± 0.000 |
| CARTA component: GBDT (numeric+rules+profile+context) | 0.9984 ± 0.0003 | 0.9956 ± 0.0008 | 52.11 ± 6.49 | 99.29 ± 0.07 | 50.42 ± 3.53 / 0.089 ± 0.041 | 99.33 ± 0.11 / 1.063 ± 0.072 | 0.005 ± 0.000 |
| CARTA component: Isolation Forest | 0.9155 ± 0.0050 | 0.8007 ± 0.0161 | 8.54 ± 1.35 | 16.03 ± 0.99 | 8.73 ± 2.00 / 0.108 ± 0.035 | 16.10 ± 1.26 / 1.067 ± 0.147 | 0.278 ± 0.010 |
| **CARTA (proposed)** | 0.9997 ± 0.0001 | 0.9996 ± 0.0001 | 98.16 ± 1.20 | 99.87 ± 0.04 | 99.15 ± 0.36 / 0.135 ± 0.041 | 99.85 ± 0.05 / 0.882 ± 0.082 | 0.002 ± 0.001 |

Recall and FPR in %. ECE is computed for every score as if it were a probability (the prior pipeline presents its fused score in [0,1] and thresholds it as such).

Paired bootstrap (1,000 resamples of the test split, per seed), CARTA − prior pipeline:

- seed 0: ΔROC-AUC = +0.0024 (95% CI +0.0019, +0.0030); ΔRecall@FPR1% = +5.81 pp (95% CI +4.88, +6.92)
- seed 1: ΔROC-AUC = +0.0022 (95% CI +0.0016, +0.0028); ΔRecall@FPR1% = +4.50 pp (95% CI +3.72, +5.47)
- seed 2: ΔROC-AUC = +0.0029 (95% CI +0.0023, +0.0036); ΔRecall@FPR1% = +5.59 pp (95% CI +4.78, +6.35)

## Table 2. Prior pipeline at its fixed operating points, 0.6/0.4 fusion with score-only levels (test split)

| Operating point | Recall (%) | FPR (%) | Precision (%) |
|---|---|---|---|
| suspicious_low_plus(final>=0.3) | 99.98 ± 0.00 | 80.85 ± 0.93 | 37.57 ± 0.27 |
| alert_medium_plus(final>=0.5) | 99.91 ± 0.04 | 37.22 ± 0.20 | 56.64 ± 0.14 |
| is_anomaly(final>=0.55) | 99.84 ± 0.11 | 16.63 ± 0.20 | 74.51 ± 0.24 |

Prior pipeline levels (cumulative, level or higher):

| Level | Score-only FPR (%) | Score-only recall (%) | Score×impact FPR (%) | Score×impact recall (%) |
|---|---|---|---|---|
| low | 80.85 ± 0.93 | 99.98 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| medium | 37.22 ± 0.20 | 99.91 ± 0.04 | 95.82 ± 0.59 | 99.98 ± 0.00 |
| high | 13.14 ± 0.09 | 99.78 ± 0.12 | 52.35 ± 0.91 | 99.94 ± 0.01 |
| critical | 0.53 ± 0.08 | 92.28 ± 1.00 | 11.43 ± 0.13 | 43.16 ± 0.42 |

## Table 3. Per-source results (test split)

| Source | n attack / benign (seed 0) | CARTA ROC-AUC | Prior ROC-AUC | CARTA R@FPR1% | Prior R@FPR1% | CARTA R@FPR0.1% | Prior R@FPR0.1% |
|---|---|---|---|---|---|---|---|
| csic2010 | 2312 / 1445 | 0.9995 ± 0.0001 | 0.9883 ± 0.0001 | 99.68 ± 0.16 | 84.01 ± 1.41 | 89.85 ± 1.71 | 61.94 ± 5.13 |
| httpparams | 1764 / 2896 | 0.9997 ± 0.0001 | 0.9993 ± 0.0001 | 99.94 ± 0.00 | 99.85 ± 0.07 | 99.92 ± 0.03 | 99.70 ± 0.14 |
| weblogs2019 | 71 / 4315 | 0.9973 ± 0.0021 | 0.9868 ± 0.0082 | 97.64 ± 0.83 | 96.22 ± 2.19 | 93.86 ± 2.22 | 81.60 ± 2.82 |
| webpayloads | 66 / 0 | n/a | n/a | n/a | n/a | n/a | n/a |

Per-source FPR thresholds are computed within the source (threshold-free metric).

Per-family recall at the validation-chosen 1%-FPR threshold:

| Family | n (seed 0) | CARTA (%) | Prior (%) |
|---|---|---|---|
| cmdi | 13 | 97.44 ± 4.44 | 58.97 ± 22.21 |
| cmdinj | 12 | 97.22 ± 4.81 | 47.22 ± 9.62 |
| csic_anomalous | 2312 | 99.90 ± 0.07 | 93.94 ± 0.35 |
| csrf | 14 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| path-traversal | 43 | 100.00 ± 0.00 | 51.16 ± 4.03 |
| rce | 2 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| scanning | 14 | 83.33 ± 4.12 | 28.57 ± 12.37 |
| sqli | 1696 | 99.94 ± 0.06 | 98.15 ± 0.59 |
| ssrf | 12 | 100.00 ± 0.00 | 63.89 ± 25.46 |
| xss | 95 | 100.00 ± 0.00 | 95.79 ± 2.11 |

## Table 4. Conformal risk tiers (calibrated on validation, evaluated on test; cumulative, tier or higher)

| Tier | Nominal budget (%) | CARTA-Mondrian FPR (%) | CARTA-Mondrian recall (%) | CARTA-Mondrian IW-recall (%) | CARTA-global FPR (%) | CARTA-global recall (%) | CARTA-global IW-recall (%) | Prior+conformal (Mondrian) FPR / recall (%) | Prior own levels (0.6/0.4, score-only) FPR / recall (%) |
|---|---|---|---|---|---|---|---|---|---|
| low | 5.00 | 5.122 ± 0.192 | 99.94 ± 0.01 | 99.94 ± 0.03 | 4.948 ± 0.357 | 99.92 ± 0.04 | 99.92 ± 0.05 | 5.091 ± 0.077 / 98.95 ± 0.08 | 80.85 ± 0.93 / 99.98 ± 0.00 |
| medium | 1.00 | 0.967 ± 0.194 | 99.70 ± 0.12 | 99.67 ± 0.15 | 0.882 ± 0.082 | 99.85 ± 0.05 | 99.83 ± 0.07 | 0.997 ± 0.088 / 93.58 ± 0.50 | 37.22 ± 0.20 / 99.91 ± 0.04 |
| high | 0.20 | 0.239 ± 0.096 | 99.07 ± 0.42 | 98.74 ± 0.56 | 0.246 ± 0.053 | 99.68 ± 0.20 | 99.61 ± 0.26 | 0.235 ± 0.047 / 86.02 ± 2.33 | 13.14 ± 0.09 / 99.78 ± 0.12 |
| critical | 0.05 | 0.050 ± 0.018 | 93.39 ± 2.53 | 90.88 ± 3.30 | 0.069 ± 0.012 | 95.61 ± 1.84 | 94.17 ± 2.37 | 0.073 ± 0.013 / 71.70 ± 5.82 | 0.53 ± 0.08 / 92.28 ± 1.00 |

IW-recall = impact-weighted recall, with the impact of each attacked endpoint computed from the prior pipeline's own released impact weights (external to CARTA).

Per-cell view (CARTA, Mondrian): benign FPR against the cell's budget, and recall per cell for Mondrian vs global thresholds:

| Tier | Cell | Pooled into | Cell budget (%) | FPR (%) | Recall, Mondrian (%) | Recall, global (%) |
|---|---|---|---|---|---|---|
| low | high | high | 8.916 ± 0.007 | 8.457 ± 0.413 | 99.94 ± 0.06 | 99.94 ± 0.06 |
| low | low | low | 2.229 ± 0.002 | 1.506 ± 0.388 | 100.00 ± 0.00 | 99.40 ± 1.03 |
| low | std | std | 4.458 ± 0.004 | 4.732 ± 0.225 | 99.95 ± 0.02 | 99.92 ± 0.00 |
| medium | high | high | 1.783 ± 0.001 | 1.871 ± 0.725 | 99.67 ± 0.24 | 99.81 ± 0.12 |
| medium | low | low | 0.446 ± 0.000 | 0.441 ± 0.144 | 90.02 ± 6.32 | 99.40 ± 1.03 |
| medium | std | std | 0.892 ± 0.001 | 0.837 ± 0.118 | 99.93 ± 0.02 | 99.89 ± 0.02 |
| high | high | high | 0.350 ± 0.000 | 0.298 ± 0.123 | 98.14 ± 0.83 | 99.47 ± 0.38 |
| high | low | std | 0.175 ± 0.000 | 0.441 ± 0.144 | 90.02 ± 6.32 | 99.40 ± 1.03 |
| high | std | std | 0.175 ± 0.000 | 0.218 ± 0.107 | 99.91 ± 0.02 | 99.84 ± 0.07 |
| critical | high | high | 0.087 ± 0.000 | 0.027 ± 0.047 | 86.05 ± 4.51 | 91.54 ± 3.12 |
| critical | low | std | 0.044 ± 0.000 | 0.441 ± 0.144 | 90.02 ± 6.32 | 97.42 ± 2.25 |
| critical | std | std | 0.044 ± 0.000 | 0.033 ± 0.016 | 98.55 ± 1.32 | 98.39 ± 1.16 |

Benign validation points per cell (seed 0): {'high': 1238, 'low': 375, 'std': 7042}; test: {'high': 1229, 'low': 390, 'std': 7037}.

Validity check: held-out data (validation + test) re-split at random into calibration and evaluation halves 200 times per seed, model fixed. Mean benign FPR on the evaluation halves vs the exact split-conformal bound (n+1-k)/(n+1), averaged over seeds:

| Tier | Budget (%) | Mondrian mean FPR (%) | Mondrian exact bound (%) | Global mean FPR (%) | Global exact bound (%) | Mondrian mean recall (%) |
|---|---|---|---|---|---|---|
| low | 5.00 | 4.910 ± 0.086 | 4.984 ± 0.001 | 4.965 ± 0.010 | 4.995 ± 0.000 | 99.94 ± 0.03 |
| medium | 1.00 | 0.973 ± 0.019 | 0.981 ± 0.001 | 0.991 ± 0.005 | 0.994 ± 0.000 | 99.68 ± 0.14 |
| high | 0.20 | 0.173 ± 0.003 | 0.190 ± 0.000 | 0.195 ± 0.002 | 0.196 ± 0.000 | 99.05 ± 0.32 |
| critical | 0.05 | 0.043 ± 0.003 | 0.046 ± 0.000 | 0.045 ± 0.003 | 0.046 ± 0.000 | 94.58 ± 2.60 |

## Table 5. Expected SOC cost per 10,000 events (threshold chosen on validation; cost on test)

| Setting | CARTA | Prior (validation-optimised threshold) | Prior at 0.5 | Prior at 0.55 | CARTA TPR / FPR (%) | Prior-opt TPR / FPR (%) |
|---|---|---|---|---|---|---|
| cfn10_prev1% | 20.3 ± 2.0 | 128.8 ± 12.7 | 3685.6 ± 20.3 | 1647.8 ± 20.4 | 99.34 ± 0.60 / 0.139 ± 0.046 | 91.12 ± 1.74 / 0.404 ± 0.050 |
| cfn100_prev1% | 52.4 ± 14.8 | 409.4 ± 11.5 | 3694.2 ± 21.6 | 1662.0 ± 30.0 | 99.75 ± 0.14 / 0.281 ± 0.007 | 98.43 ± 0.43 / 2.545 ± 0.363 |
| cfn100_prev1%_cap1.5% | 52.4 ± 14.8 | 753.2 ± 47.7 | 3694.2 ± 21.6 | 1662.0 ± 30.0 | 99.75 ± 0.14 / 0.281 ± 0.007 | 93.09 ± 0.39 / 0.632 ± 0.098 |
| cfn20_prev5% | 51.2 ± 14.8 | 399.3 ± 12.3 | 3545.3 ± 20.8 | 1595.5 ± 29.2 | 99.75 ± 0.14 / 0.281 ± 0.007 | 98.43 ± 0.43 / 2.545 ± 0.363 |

Setting cfnX_prevY%: c_FP = 1, c_FN = X, attack prevalence Y; capZ% = alerts limited to Z% of events.

Fit time (s): prior 81 ± 22, CARTA 1002 ± 450.
