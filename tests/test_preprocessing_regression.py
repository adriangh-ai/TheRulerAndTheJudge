from __future__ import annotations

import numpy as np
import pandas as pd

from lm_bmfrm.irt_utils import conformal_q, prepare_llm_judge_data


def test_score_list_becomes_one_empirical_distribution_and_records_count():
    df = pd.DataFrame(
        {
            "item_id": ["i1"],
            "system_id": ["s1"],
            "rater_id": ["j1"],
            "dimension": ["coherence"],
            "score": [3],
            "score_list": [[2, 3, 3, 5]],
        }
    )
    prepared, meta, q_obs, obs_weight = prepare_llm_judge_data(df, n_classes=5)
    np.testing.assert_array_equal(obs_weight, np.array([4.0], dtype=np.float32))
    np.testing.assert_allclose(q_obs[0], [0.0, 0.25, 0.5, 0.0, 0.25], rtol=0.0, atol=0.0)
    assert prepared.loc[0, "y_ord"] == 2
    assert meta["n_classes"] == 5


def test_conformal_quantile_uses_finite_sample_order_statistic():
    residuals = np.array([0.1, 0.2, 0.3, 0.4])
    # ceil((4+1) * .8) = 4 => maximum residual.
    assert conformal_q(residuals, alpha=0.2) == 0.4
