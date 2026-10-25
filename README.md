# CARTA: calibrated, context-aware risk tiering for cloud API attack detection

Code, result files and figure scripts for the paper

> S. S. Sefati, A. Vlahov, R. Craciunescu, B. Arasteh and V. Poulkov, *Calibrated and Context-Aware Risk Tiering for Cloud API Attack Detection Based on Stacked Learning and Mondrian Conformal Prediction*, manuscript, 2026.

The repository contains:

1. a **de-duplicated benchmark** built from the public datasets used by Abibulaiev, Pukach and Vovk
   (*Mach. Learn. Knowl. Extr.* 2026, 8, 25, <https://doi.org/10.3390/make8010025>): no canonical request
   occurs in two splits;
2. a **re-implementation of their hybrid ML/NLP pipeline** from the paper, used as the baseline (no code of
   the original release is included or copied);
3. **CARTA**, the proposed detector: fixpoint normalisation, character and structure-token text models,
   endpoint-conditional parameter profiles, cross-fitted stacking with calibration chosen on training data,
   **Mondrian split-conformal risk tiers** whose false-positive budgets depend on business context, and
   cost-constrained alert thresholds;
4. every script that produces the tables and figures of the paper, and the per-seed result files.

No dataset is redistributed here; the benchmark is rebuilt deterministically from the original files (see
[Data](#data)).

## Main results

Test split, mean ± s.d. over five seeds (12,578 test requests per seed). Full tables: [`results/RESULTS.md`](results/RESULTS.md).

| | CARTA | Prior pipeline (re-implementation, 0.6/0.4 fusion) |
|---|---|---|
| ROC-AUC | 0.9998 ± 0.0001 | 0.9976 ± 0.0003 |
| Recall at 0.1% FPR | 99.33 ± 0.34% | 83.37 ± 2.22% |
| Recall at 1% FPR | 99.89 ± 0.05% | 95.11 ± 1.06% |
| Expected calibration error | 0.001 | 0.303 |
| Recall at 0.1% FPR, GET/POST twins merged before splitting (`results_twin/`) | 98.98 ± 0.45% | 84.25 ± 2.25% |
| Recall at 0.1% FPR, strict split without the background rule (`results_strict/`, seeds 0–2) | 98.16 ± 1.20% | 75.80 ± 0.79% |
| Recall under an adaptive evasion attacker (validation 1%-FPR threshold) | 99.79% | 92.14% |

Risk tiers (budgets 5%, 1%, 0.2% and 0.05% of benign traffic): over 200 calibration/evaluation re-splits per
seed, the benign false-positive rate of every Mondrian tier, averaged over the five seeds, stays at or below its
finite-sample bound, also inside the sensitive cell (payments, authentication, administration, profile), where a single global threshold
exceeds the cell budget 2.0–3.3 times. Limits are reported as well: CARTA transfers worse than the prior
pipeline across applications (single-source training) and needs per-site recalibration (unseen-site test).

## Installation

Python 3.13 was used. Exact versions of the reported run are pinned in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest && python -m pytest -q tests        # quick checks, no data needed
```

## Data

Download the files below and put them in one folder (called `$D` here). Their MD5 hashes are those recorded in
`results/bench/manifest_seed*.json`, so you can check that you have the same files.

| Source | File used | Where | MD5 of the file used |
|---|---|---|---|
| CSIC 2010 (Kaggle CSV of the original dataset) | `csic_database.csv` | <https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks>; original: <https://www.tic.itefi.csic.es/dataset/> | `48e8abc66b8a57887fbace07a669087a` |
| HttpParamsDataset (MIT licence) | `payload_full.csv` | <https://github.com/Morzeux/HttpParamsDataset> (the Kaggle copy `evg3n1j/httpparamsdataset` is the same file) | `38ff928ee312b2b5742f306c5e83c29c` |
| Web Application Payloads | `WEB_APPLICATION_PAYLOADS.jsonl` | <https://www.kaggle.com/datasets/cyberprince/web-application-payloads-dataset> | `51b81febac3056803288982bfb22da22` |
| Web Server Access Logs – Labeled (CC0; logs of Zaker 2019, <https://doi.org/10.7910/DVN/3QBYB5>) | the downloaded zip, reduced by `prepare_zanbil.py` to `zanbil_sample.csv` | <https://www.kaggle.com/datasets/mahendradata/web-server-access-logs-labeled> | sample: `9a340e040d84031fc967b98fbff6aa63` |
| Access Log (unlabelled; unseen-site test only) | the downloaded zip | <https://www.kaggle.com/datasets/dsfelix/access-log> | – |

```bash
python prepare_zanbil.py "$D/<downloaded Web Server Access Logs - Labeled zip>" --out "$D/zanbil_sample.csv" --rate 0.06
```

The sample keeps every unique URL whose MD5 hash falls in a fixed 6% bucket (the default rate), with all its
occurrences, plus all 1,430 `sqli`, `scanning` and `rce` rows; the 1,526 `bot` rows (the log's fetches of
`/robots.txt`) are excluded. The logs come from the web server of the online shop zanbil.ir, hence the file names.

## Reproducing the paper

```bash
D=path/to/data
COMMON="--csic-csv $D/csic_database.csv --httpparams $D/payload_full.csv \
        --webpayloads $D/WEB_APPLICATION_PAYLOADS.jsonl --weblogs $D/zanbil_sample.csv"

# main protocol (benchmark, both systems, metrics, tiers, costs, evasion, ablations, single-source training)
python run_experiments.py $COMMON --out results --seeds 0 --stages main,evasion,ablation,cross --save-models
python run_experiments.py $COMMON --out results --seeds 1 2 3 4 --stages main,evasion

# risk tiers, validity checks and label noise
python posthoc_tiers.py --results results
python label_noise.py --results results
python cell_resplit.py --results results --reps 200
python split_pvalues.py --results results --reps 2000

# robustness checks
python twin_check.py --results results
python verify_normalizer.py --results results --n 30000
python versioned_comment_check.py --results results --seed 0
python latency_bench.py --results results --seed 0 --n 500        # run with nothing else on the machine; also times CARTA without its Isolation Forest
python prior_calibration_sensitivity.py --results results --seed 0  # prior: benign-row vs all-row calibration
python site_shift.py --results results --seed 0 --log "$D/<downloaded Access Log zip>" --n 20000 --rate 0.45

# strict split (no background rule) and GET/POST twins merged before splitting
python run_experiments.py $COMMON --out results_strict --seeds 0 1 2 --stages main --strict-split
python posthoc_tiers.py --results results_strict
python aggregate_results.py --results results_strict --title "Strict split (no background rule), seeds 0-2"
python run_experiments.py $COMMON --out results_twin --seeds 0 1 2 3 4 --stages main,evasion --twin-merge
python posthoc_tiers.py --results results_twin && python label_noise.py --results results_twin
python twin_check.py --results results_twin
python aggregate_results.py --results results_twin --title "Results (twin-merged benchmark)"

# tables (RESULTS.md), numbers quoted only in the text, and the vector figures of the paper
python aggregate_results.py --results results --title "CARTA vs prior pipeline: final results (5 seeds)"
python extra_numbers.py --results results          # prior operating points, reliability bins, ties, encodings, mislabelled log requests
python paper_figures.py --results results --out figures

# additional check, not reported in the paper: text-only and tabular-only stacks with and without context features
python context_ablation.py --results results --seed 0
```

The reported numbers were produced with Python 3.13.17, scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1 and
pandas 3.0.5 on a virtual machine with 2 vCPUs (Intel Xeon, 2.80 GHz) and 7 GB of memory. The test predictions
of every seed are included (`results*/preds_seed*.npz`), so `aggregate_results.py` and `paper_figures.py` run on
the files of this repository alone; the tier, label-noise and twin analyses and `extra_numbers.py` also need the
benchmark pickles, and the latency, unseen-site and comment checks the saved models, which `run_experiments.py`
writes.

## Protocol in one paragraph

Each request gets a canonical key (method plus fixpoint-normalised path, query and body; cookies and session
identifiers removed). Keys seen with both labels are dropped, one row per key is kept across all sources, keys
seen at least 20 times in the raw traffic go to training, and the rest is split 70/15/15, stratified by source,
label and attack family; zero key overlap between splits is asserted and a manifest with input hashes is
written. Models are fitted on the training split only (stacking is cross-fitted inside it), calibration and
thresholds use the validation split only, and every reported number comes from the untouched test split.
GET/POST twins (the same path and parameters sent with another method or in the body instead of the query) keep
separate canonical keys; about 20% of test requests have such a twin in training (`twin_check.py`), and
`--twin-merge` keys requests by path and parameter multiset so that twins land in one split (`results_twin/`).

## Repository layout

```
carta/
  data.py          loaders: CSIC 2010 (raw text or Kaggle CSV), HttpParamsDataset, Web Application Payloads,
                   labelled web-server log sample, Apache access logs
  normalize.py     fixpoint (encoding-invariant) normaliser, single-pass normaliser of the prior pipeline,
                   canonical request text, de-duplication key and GET/POST twin key
  benchmark.py     de-duplicated benchmark: label conflicts, cross-source de-duplication, background rule,
                   stratified splits, zero-overlap assertion, manifest
  features.py      request statistics, business context and sensitivity cells, structure tokens, generic rules
  profile.py       endpoint-conditional parameter profiles learned on benign history
  baseline.py      re-implementation of the prior pipeline (Isolation Forest, One-Class SVM, autoencoder,
                   TF-IDF models, rules, 0.6/0.4 or adaptive fusion, score-only or score x impact levels)
  proposed.py      CARTA: cross-fitted stacking, calibration chosen by inner CV, ConformalTiers (Mondrian
                   split-conformal tiers), cost-constrained thresholds
  evasion.py       obfuscation transforms on decoded parameter values and benign padding
                   (the adaptive, per-request worst-case attacker is in run_experiments.py)
  metrics.py       recall at a fixed FPR, thresholds, ECE, Brier score, confusion counts, per-group recall
run_experiments.py, posthoc_tiers.py, cell_resplit.py, split_pvalues.py, label_noise.py, twin_check.py,
verify_normalizer.py, versioned_comment_check.py, latency_bench.py, site_shift.py, prepare_zanbil.py,
prior_calibration_sensitivity.py, context_ablation.py, extra_numbers.py, aggregate_results.py, paper_figures.py
                   the experiment scripts listed above
results/           five seeds: RESULTS.md, per-seed JSON files, test predictions, corrected labels, manifests
results_strict/    strict split without the background rule (seeds 0-2)
results_twin/      GET/POST twins merged before splitting (seeds 0-4)
figures/           vector figures of the paper (paper_figures.py)
tests/             normaliser invariance and conformal-threshold checks on synthetic data
```

Notes on the result files:

- JSON keys named `prior_shipped_*` refer to the prior pipeline's own fixed thresholds (0.5 and 0.55) with
  0.6/0.4 fusion and score-only levels.
- `results/` and `results_strict/` were written before the `--twin-merge` option was added, so their manifests
  and result files have no `twin_merge` field (the option is off for both); otherwise their files have the
  structure the current code writes.
- `results/latency_carta_without_iforest_seed0.json` (second row of the latency table) is written by
  `latency_bench.py` together with `results/latency_seed0.json`.
- In `results/site_shift_seed0.json` the raw requests of the Access Log dataset are replaced by SHA-256 hashes of
  the exact strings, because that dataset states no licence; `site_shift.py` writes the file in this form and
  prints the top-ranked requests when run on the downloaded file.
- `results/site_shift_top100_annotation_seed0.csv` is the manual inspection behind the statement that the
  top-ranked requests of the unseen site are attacks: the 100 top-ranked requests of each detector, labelled
  from the signature list of `label_noise.py` where one matches (`evidence` = `signature:...`) and by hand
  otherwise (`evidence` = `manual:...`). `site_shift.py --annotation-template` writes the sheet with the
  signature rows filled in.
- `results*/label_noise_seed*.json` list the benign-labelled log requests with unambiguous attack signatures that
  are counted as attacks in the corrected-label analysis (from the CC0 Web Server Access Logs – Labeled dataset).
  `extra_numbers.py` also counts such requests over the whole benchmark (key `log_signature_requests_benchmark`,
  added after `results/extra_numbers.json` was written; re-run `extra_numbers.py` to add it).
- Benchmark pickles (`results*/bench/bench_seed*.pkl`, several hundred MB) and fitted models are not included;
  `run_experiments.py` rebuilds them.

## Known limitation of the normaliser

The normaliser deletes `/*...*/` comments with their content, although MySQL executes the content of
`/*!...*/` comments. `versioned_comment_check.py` measures the effect (on seed 0, recall stays at 100% at the
validation 1%-FPR threshold and 99.94% at the 0.1% threshold); the content of executable comments should be kept
in a future version. Such comments do not occur in the benchmark, but five of the top-ranked requests of the
unseen site use them. The reported results use the normaliser as it is in this repository.

## Citation

```bibtex
@unpublished{sefati2026carta,
  author = {Sefati, Seyed Salar and Vlahov, Atanas and Craciunescu, Razvan and Arasteh, Bahman and Poulkov, Vladimir},
  title  = {Calibrated and Context-Aware Risk Tiering for Cloud {API} Attack Detection Based on Stacked Learning and {Mondrian} Conformal Prediction},
  note   = {Manuscript},
  year   = {2026}
}
```

Please also cite the datasets you use (see [Data](#data)) and, for the baseline, Abibulaiev et al. (2026).

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). The datasets are not part of
this repository and keep their own terms.
