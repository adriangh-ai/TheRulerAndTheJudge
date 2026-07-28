from __future__ import annotations
import numpy as np
from lm_bmfrm.evaluation_metrics import (
    compute_false_separation_rate,
    pairwise_dominance_probability,
)
from lm_bmfrm.estimate import ordered_probs_from_eta

def test_fsr_denominator_is_human_indistinguishable_pairs():
    human = {("a", "b"): 0.55, ("a", "c"): 0.90, ("b", "c"): 0.45}
    judge = {("a", "b"): 0.95, ("a", "c"): 0.99, ("b", "c"): 0.50}
    # Human-indistinguishable pairs are (a,b) and (b,c); only (a,b) is falsely separated.
    assert compute_false_separation_rate(human, judge, tau_h=0.8, tau_l=0.8) == 0.5

def test_ordered_probabilities_are_normalized():
    eta = np.array([-2.0, 0.0, 2.0])
    cuts = np.array([-1.5, -0.25, 0.75, 1.75])
    probs = ordered_probs_from_eta(eta, cuts)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)
    assert np.all(probs > 0.0)

def test_pairwise_dominance_uses_mid_ties():
    grid = np.array([-1.0, 0.0, 1.0])
    high = np.array([0.0, 0.0, 1.0])
    middle = np.array([0.0, 1.0, 0.0])

    assert pairwise_dominance_probability(high, middle, grid) == 1.0
    assert pairwise_dominance_probability(middle, middle, grid) == 0.5


def test_pairwise_dominance_is_complementary_on_shared_grid():
    grid = np.array([-1.0, 0.0, 1.0])
    a = np.array([0.2, 0.5, 0.3])
    b = np.array([0.6, 0.1, 0.3])
    p_ab = pairwise_dominance_probability(a, b, grid)
    p_ba = pairwise_dominance_probability(b, a, grid)
    assert p_ab + p_ba == 1.0
