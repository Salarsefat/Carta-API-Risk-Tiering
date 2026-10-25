"""Finite-sample behaviour of the split-conformal thresholds and the Mondrian tiers (carta/proposed.py)."""
import math

import numpy as np

from carta.proposed import ConformalTiers, conformal_threshold


def test_threshold_is_the_conformal_order_statistic():
    s = np.arange(1, 100, dtype=float)  # n = 99 distinct scores
    eps = 0.05
    k = math.ceil((99 + 1) * (1 - eps))  # 95
    assert conformal_threshold(s, eps) == s[k - 1]


def test_uncertifiable_budget_gives_infinite_threshold():
    assert math.isinf(conformal_threshold(np.random.default_rng(0).random(10), 0.01))


def test_false_positive_rate_matches_the_exact_bound_on_exchangeable_scores():
    rng = np.random.default_rng(1)
    n, eps, reps = 400, 0.05, 4000
    k = math.ceil((n + 1) * (1 - eps))
    bound = (n + 1 - k) / (n + 1)
    hits = 0
    for _ in range(reps):
        s = rng.random(n + 1)
        hits += s[-1] > conformal_threshold(s[:-1], eps)
    rate = hits / reps
    se = math.sqrt(bound * (1 - bound) / reps)
    assert rate <= bound + 4 * se
    assert rate >= bound - 4 * se


def test_mondrian_tiers_are_nested_and_respect_cell_budgets():
    rng = np.random.default_rng(2)
    n = 6000
    cells = rng.choice(np.array(["high", "std", "low"], dtype=object), size=n, p=[0.2, 0.6, 0.2])
    y = (rng.random(n) < 0.3).astype(int)
    p = np.where(y == 1, rng.beta(5, 1, n), rng.beta(1, 5, n))
    tiers = ConformalTiers(mondrian=True).fit(p, y, cells)
    thr = tiers.thresholds(cells)
    assert np.all(np.diff(thr, axis=1) >= 0)  # low <= medium <= high <= critical
    # fresh benign data from the same distribution: per-cell rate of "low or above" close to the budget or below
    m = 200_000
    c2 = rng.choice(np.array(["high", "std", "low"], dtype=object), size=m, p=[0.2, 0.6, 0.2])
    p2 = rng.beta(1, 5, m)
    t2 = tiers.transform(p2, c2)
    budget = tiers.cell_budget(c2, "low")
    for c in ("high", "std", "low"):
        sel = c2 == c
        rate = np.mean(t2[sel] != "normal")
        assert rate <= budget[sel][0] + 0.02  # finite-sample slack for one calibration draw
