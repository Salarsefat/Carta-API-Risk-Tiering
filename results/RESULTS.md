# CARTA vs prior pipeline: final results (5 seeds)

Seeds: [0, 1, 2, 3, 4]. Test split (seed 0): 12578 unique requests (4208 attacks, 8370 benign); train 60637, validation 12577. Strict split: False. Values are mean ± s.d. over seeds; all thresholds are chosen on the validation split and applied unchanged to the test split.

Base-calibration chosen by inner CV on training OOF scores: ['isotonic', 'isotonic', 'isotonic', 'isotonic', 'isotonic'].

## Table 1. Detection on the held-out test split

| Method | ROC-AUC | PR-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | Test recall / FPR at val-0.1% thr. | Test recall / FPR at val-1% thr. | ECE |
|---|---|---|---|---|---|---|---|
| Prior pipeline (α=0.6, β=0.4) | 0.9976 ± 0.0003 | 0.9953 ± 0.0008 | 83.37 ± 2.22 | 95.11 ± 1.06 | 85.49 ± 2.63 / 0.131 ± 0.035 | 95.13 ± 0.92 / 1.018 ± 0.228 | 0.303 ± 0.002 |
| Prior pipeline (adaptive fusion) | 0.9955 ± 0.0008 | 0.9915 ± 0.0014 | 79.61 ± 4.53 | 92.40 ± 1.06 | 79.04 ± 3.31 / 0.108 ± 0.034 | 92.33 ± 0.75 / 1.008 ± 0.159 | 0.320 ± 0.003 |
| Prior ML branch (IF/OC-SVM/AE, max) | 0.9189 ± 0.0058 | 0.8433 ± 0.0117 | 9.11 ± 1.60 | 32.92 ± 0.89 | 9.29 ± 1.82 / 0.093 ± 0.034 | 32.67 ± 0.57 / 0.946 ± 0.107 | 0.382 ± 0.002 |
| Prior NLP branch (TF-IDF LR/char SGD/rules, max) | 0.9986 ± 0.0002 | 0.9972 ± 0.0005 | 93.26 ± 0.70 | 98.12 ± 0.26 | 93.84 ± 0.42 / 0.112 ± 0.032 | 98.25 ± 0.28 / 1.061 ± 0.170 | 0.100 ± 0.001 |
| Rules only | 0.7249 ± 0.0039 | 0.6638 ± 0.0043 | 40.23 ± 2.19 | 44.74 ± 0.58 | 41.35 ± 2.92 / 0.055 ± 0.066 | 44.74 ± 0.58 / 0.103 ± 0.054 | 0.202 ± 0.003 |
| CARTA component: char n-gram LR | 0.9988 ± 0.0003 | 0.9980 ± 0.0003 | 95.40 ± 1.34 | 98.58 ± 0.42 | 96.02 ± 1.21 / 0.143 ± 0.040 | 98.64 ± 0.33 / 1.051 ± 0.136 | 0.043 ± 0.002 |
| CARTA component: structure-token LR | 0.9991 ± 0.0001 | 0.9986 ± 0.0003 | 96.78 ± 1.47 | 99.32 ± 0.09 | 97.15 ± 0.95 / 0.122 ± 0.054 | 99.31 ± 0.12 / 0.999 ± 0.070 | 0.034 ± 0.001 |
| CARTA component: GBDT (numeric+rules+profile+context) | 0.9984 ± 0.0002 | 0.9960 ± 0.0005 | 54.90 ± 3.58 | 99.41 ± 0.18 | 51.02 ± 17.42 / 0.091 ± 0.038 | 99.43 ± 0.22 / 1.008 ± 0.155 | 0.006 ± 0.001 |
| CARTA component: Isolation Forest | 0.9150 ± 0.0076 | 0.8052 ± 0.0180 | 8.13 ± 0.99 | 16.12 ± 2.21 | 7.79 ± 0.82 / 0.081 ± 0.029 | 16.17 ± 2.42 / 1.004 ± 0.072 | 0.284 ± 0.013 |
| **CARTA (proposed)** | 0.9998 ± 0.0001 | 0.9997 ± 0.0001 | 99.33 ± 0.34 | 99.89 ± 0.05 | 99.54 ± 0.14 / 0.131 ± 0.044 | 99.88 ± 0.05 / 1.051 ± 0.136 | 0.001 ± 0.000 |

Recall and FPR in %. ECE is computed for every score as if it were a probability (the prior pipeline presents its fused score in [0,1] and thresholds it as such).

Paired bootstrap (1,000 resamples of the test split, per seed), CARTA − prior pipeline:

- seed 0: ΔROC-AUC = +0.0019 (95% CI +0.0014, +0.0025); ΔRecall@FPR1% = +3.56 pp (95% CI +2.63, +5.09)
- seed 1: ΔROC-AUC = +0.0021 (95% CI +0.0016, +0.0028); ΔRecall@FPR1% = +4.53 pp (95% CI +3.64, +5.36)
- seed 2: ΔROC-AUC = +0.0029 (95% CI +0.0023, +0.0035); ΔRecall@FPR1% = +6.40 pp (95% CI +5.50, +7.38)
- seed 3: ΔROC-AUC = +0.0021 (95% CI +0.0015, +0.0028); ΔRecall@FPR1% = +4.64 pp (95% CI +3.98, +5.33)
- seed 4: ΔROC-AUC = +0.0016 (95% CI +0.0012, +0.0021); ΔRecall@FPR1% = +4.76 pp (95% CI +3.93, +5.62)

## Table 2. Prior pipeline at its fixed operating points, 0.6/0.4 fusion with score-only levels (test split)

| Operating point | Recall (%) | FPR (%) | Precision (%) |
|---|---|---|---|
| suspicious_low_plus(final>=0.3) | 100.00 ± 0.00 | 84.38 ± 0.62 | 37.34 ± 0.17 |
| alert_medium_plus(final>=0.5) | 99.92 ± 0.03 | 38.79 ± 0.37 | 56.43 ± 0.23 |
| is_anomaly(final>=0.55) | 99.89 ± 0.05 | 17.11 ± 0.38 | 74.59 ± 0.42 |

Prior pipeline levels (cumulative, level or higher):

| Level | Score-only FPR (%) | Score-only recall (%) | Score×impact FPR (%) | Score×impact recall (%) |
|---|---|---|---|---|
| low | 84.38 ± 0.62 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| medium | 38.79 ± 0.37 | 99.92 ± 0.03 | 97.66 ± 0.36 | 100.00 ± 0.00 |
| high | 13.65 ± 0.50 | 99.82 ± 0.04 | 53.04 ± 0.38 | 99.94 ± 0.03 |
| critical | 0.50 ± 0.16 | 93.18 ± 0.51 | 11.80 ± 0.41 | 43.44 ± 0.24 |

## Table 3. Per-source results (test split)

| Source | n attack / benign (seed 0) | CARTA ROC-AUC | Prior ROC-AUC | CARTA R@FPR1% | Prior R@FPR1% | CARTA R@FPR0.1% | Prior R@FPR0.1% |
|---|---|---|---|---|---|---|---|
| csic2010 | 2307 / 1442 | 0.9998 ± 0.0002 | 0.9889 ± 0.0022 | 99.71 ± 0.22 | 85.20 ± 2.68 | 95.87 ± 5.19 | 69.02 ± 1.98 |
| httpparams | 1765 / 2896 | 0.9999 ± 0.0002 | 0.9996 ± 0.0002 | 99.97 ± 0.05 | 99.89 ± 0.04 | 99.93 ± 0.07 | 99.73 ± 0.12 |
| weblogs2019 | 69 / 4032 | 0.9970 ± 0.0057 | 0.9928 ± 0.0063 | 98.55 ± 1.45 | 97.39 ± 0.65 | 95.07 ± 2.20 | 87.25 ± 3.30 |
| webpayloads | 67 / 0 | n/a | n/a | n/a | n/a | n/a | n/a |

Per-source FPR thresholds are computed within the source (threshold-free metric).

Per-family recall at the validation-chosen 1%-FPR threshold:

| Family | n (seed 0) | CARTA (%) | Prior (%) |
|---|---|---|---|
| cmdi | 13 | 95.38 ± 4.21 | 41.54 ± 11.67 |
| cmdinj | 12 | 98.33 ± 3.73 | 56.67 ± 16.03 |
| csic_anomalous | 2307 | 99.93 ± 0.08 | 94.38 ± 1.16 |
| csrf | 15 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| path-traversal | 44 | 99.55 ± 1.02 | 51.82 ± 13.67 |
| rce | 1 | 100.00 ± 0.00 | 80.00 ± 44.72 |
| scanning | 13 | 84.62 ± 9.42 | 26.15 ± 6.88 |
| sqli | 1696 | 99.98 ± 0.03 | 98.58 ± 0.99 |
| ssrf | 12 | 100.00 ± 0.00 | 65.00 ± 19.90 |
| xss | 95 | 100.00 ± 0.00 | 96.63 ± 1.73 |

## Table 4. Conformal risk tiers (calibrated on validation, evaluated on test; cumulative, tier or higher)

| Tier | Nominal budget (%) | CARTA-Mondrian FPR (%) | CARTA-Mondrian recall (%) | CARTA-Mondrian IW-recall (%) | CARTA-global FPR (%) | CARTA-global recall (%) | CARTA-global IW-recall (%) | Prior+conformal (Mondrian) FPR / recall (%) | Prior own levels (0.6/0.4, score-only) FPR / recall (%) |
|---|---|---|---|---|---|---|---|---|---|
| low | 5.00 | 4.583 ± 0.491 | 99.96 ± 0.04 | 99.95 ± 0.05 | 4.898 ± 0.361 | 99.96 ± 0.04 | 99.96 ± 0.03 | 5.063 ± 0.169 / 98.74 ± 0.75 | 84.38 ± 0.62 / 100.00 ± 0.00 |
| medium | 1.00 | 1.080 ± 0.091 | 99.86 ± 0.09 | 99.82 ± 0.14 | 1.051 ± 0.136 | 99.88 ± 0.05 | 99.88 ± 0.06 | 0.977 ± 0.184 / 94.01 ± 0.98 | 38.79 ± 0.37 / 99.92 ± 0.03 |
| high | 0.20 | 0.239 ± 0.064 | 99.36 ± 0.41 | 99.10 ± 0.60 | 0.251 ± 0.025 | 99.73 ± 0.14 | 99.68 ± 0.20 | 0.203 ± 0.049 / 85.81 ± 1.82 | 13.65 ± 0.50 / 99.82 ± 0.04 |
| critical | 0.05 | 0.072 ± 0.047 | 96.00 ± 3.15 | 94.66 ± 4.31 | 0.081 ± 0.077 | 98.48 ± 1.33 | 98.04 ± 1.66 | 0.067 ± 0.040 / 74.73 ± 3.46 | 0.50 ± 0.16 / 93.18 ± 0.51 |

IW-recall = impact-weighted recall, with the impact of each attacked endpoint computed from the prior pipeline's own released impact weights (external to CARTA).

Per-cell view (CARTA, Mondrian): benign FPR against the cell's budget, and recall per cell for Mondrian vs global thresholds:

| Tier | Cell | Pooled into | Cell budget (%) | FPR (%) | Recall, Mondrian (%) | Recall, global (%) |
|---|---|---|---|---|---|---|
| low | high | high | 8.873 ± 0.011 | 8.899 ± 0.666 | 99.94 ± 0.07 | 99.99 ± 0.03 |
| low | low | low | 2.218 ± 0.003 | 1.289 ± 0.558 | 100.00 ± 0.00 | 99.23 ± 1.06 |
| low | std | std | 4.437 ± 0.006 | 3.970 ± 0.622 | 99.98 ± 0.02 | 99.95 ± 0.05 |
| medium | high | high | 1.775 ± 0.002 | 1.966 ± 0.534 | 99.74 ± 0.24 | 99.91 ± 0.11 |
| medium | low | low | 0.444 ± 0.001 | 0.106 ± 0.238 | 99.58 ± 0.93 | 99.23 ± 1.06 |
| medium | std | std | 0.887 ± 0.001 | 0.970 ± 0.120 | 99.94 ± 0.02 | 99.88 ± 0.04 |
| high | high | high | 0.348 ± 0.000 | 0.420 ± 0.253 | 98.62 ± 0.98 | 99.60 ± 0.32 |
| high | low | std | 0.174 ± 0.000 | 0.000 ± 0.000 | 99.23 ± 1.06 | 99.23 ± 1.06 |
| high | std | std | 0.174 ± 0.000 | 0.219 ± 0.073 | 99.88 ± 0.04 | 99.83 ± 0.02 |
| critical | high | high | 0.087 ± 0.000 | 0.129 ± 0.110 | 92.22 ± 7.17 | 97.32 ± 2.18 |
| critical | low | std | 0.044 ± 0.000 | 0.000 ± 0.000 | 79.64 ± 44.53 | 95.06 ± 10.08 |
| critical | std | std | 0.044 ± 0.000 | 0.065 ± 0.043 | 98.96 ± 1.89 | 99.37 ± 0.56 |

Benign validation points per cell (seed 0): {'high': 1252, 'low': 347, 'std': 6769}; test: {'high': 1228, 'low': 372, 'std': 6770}.

Validity check: held-out data (validation + test) re-split at random into calibration and evaluation halves 200 times per seed, model fixed. Mean benign FPR on the evaluation halves vs the exact split-conformal bound (n+1-k)/(n+1), averaged over seeds:

| Tier | Budget (%) | Mondrian mean FPR (%) | Mondrian exact bound (%) | Global mean FPR (%) | Global exact bound (%) | Mondrian mean recall (%) |
|---|---|---|---|---|---|---|
| low | 5.00 | 4.559 ± 0.403 | 4.983 ± 0.003 | 4.999 ± 0.027 | 4.994 ± 0.000 | 99.93 ± 0.05 |
| medium | 1.00 | 0.966 ± 0.014 | 0.980 ± 0.001 | 1.000 ± 0.012 | 0.994 ± 0.000 | 99.83 ± 0.09 |
| high | 0.20 | 0.187 ± 0.004 | 0.191 ± 0.000 | 0.195 ± 0.002 | 0.191 ± 0.000 | 99.26 ± 0.25 |
| critical | 0.05 | 0.048 ± 0.002 | 0.048 ± 0.000 | 0.048 ± 0.003 | 0.048 ± 0.000 | 91.93 ± 5.47 |

## Table 5. Expected SOC cost per 10,000 events (threshold chosen on validation; cost on test)

| Setting | CARTA | Prior (validation-optimised threshold) | Prior at 0.5 | Prior at 0.55 | CARTA TPR / FPR (%) | Prior-opt TPR / FPR (%) |
|---|---|---|---|---|---|---|
| cfn10_prev1% | 17.1 ± 2.9 | 117.4 ± 12.5 | 3841.3 ± 36.8 | 1695.1 ± 37.4 | 99.21 ± 0.29 / 0.093 ± 0.033 | 92.82 ± 1.02 / 0.461 ± 0.080 |
| cfn100_prev1% | 50.7 ± 13.3 | 392.3 ± 32.3 | 3848.1 ± 36.4 | 1705.4 ± 35.3 | 99.77 ± 0.10 / 0.282 ± 0.089 | 98.75 ± 0.24 / 2.695 ± 0.489 |
| cfn100_prev1%_cap1.5% | 50.7 ± 13.3 | 684.6 ± 98.1 | 3848.1 ± 36.4 | 1705.4 ± 35.3 | 99.77 ± 0.10 / 0.282 ± 0.089 | 93.76 ± 0.99 / 0.616 ± 0.097 |
| cfn20_prev5% | 49.6 ± 13.1 | 381.5 ± 30.6 | 3693.0 ± 35.0 | 1637.0 ± 33.8 | 99.77 ± 0.10 / 0.282 ± 0.089 | 98.75 ± 0.24 / 2.695 ± 0.489 |

Setting cfnX_prevY%: c_FP = 1, c_FN = X, attack prevalence Y; capZ% = alerts limited to Z% of events.

## Table 6. Evasion: attack recall at the clean validation-1%-FPR threshold

| Transform | CARTA (%) | Prior (%) | Requests whose prior-view text changed (%) |
|---|---|---|---|
| clean | 99.88 ± 0.05 | 95.13 ± 0.92 | n/a |
| double_url_encode | 99.92 ± 0.04 | 97.94 ± 0.38 | 91.1 ± 0.1 |
| mixed_case | 99.88 ± 0.05 | 95.51 ± 0.72 | 0.0 ± 0.0 |
| inline_comments | 99.84 ± 0.10 | 96.48 ± 0.48 | 79.8 ± 0.4 |
| fullwidth_unicode | 99.87 ± 0.06 | 96.04 ± 0.70 | 0.0 ± 0.0 |
| whitespace_variants | 99.87 ± 0.05 | 95.56 ± 0.70 | 51.8 ± 0.7 |
| html_entities | 99.88 ± 0.06 | 96.09 ± 0.64 | 0.0 ± 0.0 |
| combined | 99.92 ± 0.04 | 97.95 ± 0.38 | 91.1 ± 0.1 |
| benign_padding | 99.92 ± 0.03 | 95.41 ± 0.59 | 95.5 ± 0.4 |
| adaptive_worst_case | 99.79 ± 0.10 | 92.14 ± 1.04 | n/a |

adaptive_worst_case: per request, the attacker applies whichever transform (or none) minimises the score of the detector under attack (black-box, per-detector).

## Table 7. Ablations

| Variant | ROC-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | ECE | Medium-tier FPR / recall (%) |
|---|---|---|---|---|---|
| full | 0.9998 | 99.33 | 99.88 | 0.001 | 1.039 / 99.83 |
| base_calibration_none | 0.9997 | 98.36 | 99.90 | 0.002 | 0.944 / 99.83 |
| base_calibration_isotonic | 0.9998 | 99.33 | 99.88 | 0.001 | 1.039 / 99.83 |
| no_context_in_meta | 0.9997 | 99.14 | 99.86 | 0.002 | 0.968 / 99.88 |
| drop_char | 0.9996 | 98.93 | 99.83 | 0.002 | 1.063 / 99.76 |
| drop_struct | 0.9998 | 98.91 | 99.86 | 0.002 | 1.051 / 99.81 |
| drop_gbdt | 0.9994 | 95.98 | 99.60 | 0.003 | 1.171 / 99.07 |
| drop_iforest | 0.9998 | 99.45 | 99.88 | 0.001 | 1.004 / 99.86 |
| text_only(char+struct) | 0.9992 | 96.46 | 99.33 | 0.002 | 0.514 / 99.07 |
| tabular_only(gbdt+iforest) | 0.9992 | 97.31 | 99.29 | 0.003 | 1.111 / 99.31 |
| no_endpoint_profile | 0.9997 | 97.86 | 99.74 | 0.002 | 1.422 / 99.48 |
| no_context_anywhere | 0.9997 | 99.38 | 99.88 | 0.001 | 0.908 / 99.86 |

## Table 8. Single-source training (train on one source, test on each source's test split)

| Train → test | CARTA ROC-AUC | Prior ROC-AUC | CARTA R@FPR1% | Prior R@FPR1% |
|---|---|---|---|---|
| csic2010->csic2010 | 0.9999 | 0.9902 | 99.87 | 85.96 |
| csic2010->httpparams | 0.9279 | 0.9220 | 71.16 | 72.41 |
| csic2010->weblogs2019 | 0.4782 | 0.8189 | 1.45 | 68.12 |
| httpparams->csic2010 | 0.6228 | 0.6116 | 17.73 | 12.05 |
| httpparams->httpparams | 1.0000 | 0.9996 | 100.00 | 99.89 |
| httpparams->weblogs2019 | 0.7378 | 0.8828 | 21.74 | 68.12 |
| weblogs2019->csic2010 | 0.5029 | 0.6005 | 2.34 | 17.95 |
| weblogs2019->httpparams | 0.5312 | 0.8725 | 34.05 | 81.81 |
| weblogs2019->weblogs2019 | 0.9987 | 0.9995 | 98.55 | 100.00 |

Fit time (s): prior 72 ± 2, CARTA 907 ± 235.

## Table 10. Per-cell validity of the tiers over 200 re-splits per seed

Validation + test pooled and redrawn into calibration and evaluation halves (model fixed; the same re-splits for both modes). Per seed, the mean benign FPR inside each sensitivity cell over the re-splits; the table gives mean ± s.d. of that per-seed mean over seeds. Intended budget = eps_t * m_c of the cell (its pooled cell where the partition pools it); exact bound = (n_c + 1 - k) / (n_c + 1) of the Mondrian threshold. 'corrected' = benign-labelled log URLs with unambiguous attack signatures counted as attacks in both validation and test (pooled relabelled per cell, seed 0: {'high': 3, 'std': 3, 'low': 0}; pooled benign per cell: {'high': 2480, 'std': 13539, 'low': 719}).

| Labels | Tier | Cell | Intended budget (%) | Mondrian FPR (%) | Mondrian exact bound (%) | Global FPR (%) | Global FPR / budget | Mondrian recall (%) | Global recall (%) |
|---|---|---|---|---|---|---|---|---|---|
| original | low | high | 8.877 ± 0.003 | 8.810 ± 0.166 | 8.838 ± 0.003 | 29.232 ± 1.846 | 3.29 ± 0.21 | 99.91 ± 0.12 | 99.99 ± 0.01 |
| original | low | std | 4.439 ± 0.002 | 3.967 ± 0.512 | 4.431 ± 0.002 | 0.790 ± 0.345 | 0.18 ± 0.08 | 99.94 ± 0.04 | 99.92 ± 0.04 |
| original | low | low | 2.219 ± 0.001 | 1.071 ± 0.238 | 2.094 ± 0.006 | 0.000 ± 0.000 | 0.00 ± 0.00 | 99.81 ± 0.43 | 99.18 ± 0.48 |
| original | medium | high | 1.775 ± 0.001 | 1.725 ± 0.040 | 1.736 ± 0.003 | 5.505 ± 0.585 | 3.10 ± 0.33 | 99.71 ± 0.25 | 99.88 ± 0.12 |
| original | medium | std | 0.888 ± 0.000 | 0.865 ± 0.013 | 0.880 ± 0.000 | 0.206 ± 0.109 | 0.23 ± 0.12 | 99.92 ± 0.04 | 99.85 ± 0.02 |
| original | medium | low | 0.444 ± 0.000 | 0.260 ± 0.016 | 0.268 ± 0.006 | 0.000 ± 0.000 | 0.00 ± 0.00 | 99.61 ± 0.53 | 99.18 ± 0.48 |
| original | high | high | 0.348 ± 0.000 | 0.315 ± 0.010 | 0.321 ± 0.001 | 0.897 ± 0.128 | 2.58 ± 0.37 | 98.43 ± 0.64 | 99.40 ± 0.32 |
| original | high | std | 0.174 ± 0.000 | 0.173 ± 0.004 | 0.168 ± 0.000 | 0.069 ± 0.026 | 0.40 ± 0.15 | 99.85 ± 0.02 | 99.81 ± 0.03 |
| original | high | low | 0.174 ± 0.000 | 0.000 ± 0.000 | 0.168 ± 0.000 | 0.000 ± 0.000 | 0.00 ± 0.00 | 99.18 ± 0.48 | 99.18 ± 0.48 |
| original | critical | high | 0.087 ± 0.000 | 0.080 ± 0.009 | 0.080 ± 0.000 | 0.171 ± 0.072 | 1.97 ± 0.82 | 82.24 ± 12.27 | 94.38 ± 4.32 |
| original | critical | std | 0.044 ± 0.000 | 0.044 ± 0.002 | 0.042 ± 0.000 | 0.027 ± 0.014 | 0.62 ± 0.33 | 98.88 ± 0.99 | 98.82 ± 0.82 |
| original | critical | low | 0.044 ± 0.000 | 0.000 ± 0.000 | 0.042 ± 0.000 | 0.000 ± 0.000 | 0.00 ± 0.00 | 83.02 ± 19.30 | 84.71 ± 14.94 |
| corrected | low | high | 8.878 ± 0.003 | 8.814 ± 0.166 | 8.839 ± 0.004 | 29.313 ± 1.832 | 3.30 ± 0.21 | 99.87 ± 0.10 | 99.96 ± 0.03 |
| corrected | low | std | 4.439 ± 0.002 | 3.968 ± 0.519 | 4.431 ± 0.002 | 0.778 ± 0.344 | 0.18 ± 0.08 | 99.93 ± 0.05 | 99.91 ± 0.06 |
| corrected | low | low | 2.220 ± 0.001 | 1.071 ± 0.238 | 2.094 ± 0.006 | 0.000 ± 0.000 | 0.00 ± 0.00 | 99.81 ± 0.43 | 99.18 ± 0.48 |
| corrected | medium | high | 1.776 ± 0.001 | 1.728 ± 0.034 | 1.736 ± 0.002 | 5.576 ± 0.587 | 3.14 ± 0.33 | 99.68 ± 0.23 | 99.85 ± 0.10 |
| corrected | medium | std | 0.888 ± 0.000 | 0.864 ± 0.015 | 0.880 ± 0.000 | 0.194 ± 0.107 | 0.22 ± 0.12 | 99.91 ± 0.05 | 99.84 ± 0.02 |
| corrected | medium | low | 0.444 ± 0.000 | 0.260 ± 0.016 | 0.268 ± 0.006 | 0.000 ± 0.000 | 0.00 ± 0.00 | 99.61 ± 0.53 | 99.18 ± 0.48 |
| corrected | high | high | 0.348 ± 0.000 | 0.319 ± 0.010 | 0.322 ± 0.001 | 0.953 ± 0.135 | 2.74 ± 0.39 | 98.48 ± 0.67 | 99.43 ± 0.29 |
| corrected | high | std | 0.174 ± 0.000 | 0.173 ± 0.005 | 0.168 ± 0.000 | 0.059 ± 0.024 | 0.34 ± 0.14 | 99.83 ± 0.02 | 99.79 ± 0.03 |
| corrected | high | low | 0.174 ± 0.000 | 0.000 ± 0.000 | 0.168 ± 0.000 | 0.000 ± 0.000 | 0.00 ± 0.00 | 99.18 ± 0.48 | 99.18 ± 0.48 |
| corrected | critical | high | 0.087 ± 0.000 | 0.084 ± 0.009 | 0.080 ± 0.000 | 0.216 ± 0.090 | 2.48 ± 1.03 | 93.47 ± 5.72 | 98.14 ± 0.51 |
| corrected | critical | std | 0.044 ± 0.000 | 0.044 ± 0.001 | 0.042 ± 0.000 | 0.019 ± 0.016 | 0.44 ± 0.37 | 99.75 ± 0.07 | 99.63 ± 0.10 |
| corrected | critical | low | 0.044 ± 0.000 | 0.000 ± 0.000 | 0.042 ± 0.000 | 0.000 ± 0.000 | 0.00 ± 0.00 | 98.82 ± 0.84 | 99.00 ± 0.60 |

Prior pipeline's score under the same tiers (original labels), sensitive cell ('high'):

| Tier | Intended budget (%) | Mondrian FPR (%) | Mondrian recall (%) | Global FPR (%) | Global recall (%) |
|---|---|---|---|---|---|
| low | 8.877 ± 0.003 | 8.876 ± 0.105 | 97.01 ± 1.83 | 20.684 ± 0.791 | 99.35 ± 0.17 |
| medium | 1.775 ± 0.001 | 1.710 ± 0.045 | 88.58 ± 2.10 | 3.047 ± 0.531 | 92.09 ± 1.83 |
| high | 0.348 ± 0.000 | 0.311 ± 0.021 | 73.91 ± 4.40 | 0.682 ± 0.163 | 81.57 ± 2.25 |
| critical | 0.087 ± 0.000 | 0.083 ± 0.010 | 55.56 ± 2.43 | 0.173 ± 0.070 | 66.58 ± 3.12 |

