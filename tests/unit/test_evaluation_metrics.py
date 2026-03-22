from __future__ import annotations

import numpy as np
import pandas as pd

from f1_predictor.evaluation import F1ModelEvaluator


def test_dcg_and_ndcg_are_reasonable() -> None:
    ev = F1ModelEvaluator()

    rel_true = np.array([3.0, 2.0, 1.0, 0.0])
    rel_pred_perfect = np.array([3.0, 2.0, 1.0, 0.0])
    rel_pred_worst = np.array([0.0, 1.0, 2.0, 3.0])

    # Perfect ordering should yield NDCG ~ 1.0.
    ndcg_perfect = ev._ndcg_at_k(rel_pred_perfect, rel_true, k=4)
    assert 0.999 <= ndcg_perfect <= 1.001

    # Reversed ordering should be worse than perfect.
    ndcg_worst = ev._ndcg_at_k(rel_pred_worst, rel_true, k=4)
    assert ndcg_worst < ndcg_perfect


def test_map_at_k_basic() -> None:
    pred_order = np.array([2, 0, 1, 3])  # predicted ranking (0 is best)
    true_top = {0, 2}
    m = F1ModelEvaluator._map_at_k(pred_order, true_top_indices=true_top, k=3)
    # Hits at ranks 1 and 2 => AP = (1/1 + 2/2) / 2 = 1.0
    assert 0.999 <= m <= 1.001


def test_calculate_f1_specific_metrics_smoke() -> None:
    ev = F1ModelEvaluator()

    y_true = np.array([1.0, 2.0, 10.0, 20.0])
    y_pred = np.array([1.0, 5.0, 8.0, 18.0])
    m = ev._calculate_f1_specific_metrics(y_true, y_pred)

    assert "top3_accuracy" in m
    assert "top5_accuracy" in m
    assert "rank_correlation" in m
    assert 0.0 <= float(m["top3_accuracy"]) <= 1.0
    assert 0.0 <= float(m["top5_accuracy"]) <= 1.0


def test_event_level_metrics_returns_aggregates() -> None:
    ev = F1ModelEvaluator()
    X = pd.DataFrame(
        {
            "Year": [2024] * 4 + [2024] * 4,
            "Race_Num": [1] * 4 + [2] * 4,
        }
    )
    y_true = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    y_pred = np.array([1, 2, 4, 3, 2, 1, 4, 3], dtype=float)

    agg, per_event = ev._calculate_event_level_metrics(X, y_true, y_pred)
    assert isinstance(agg, dict)
    assert isinstance(per_event, pd.DataFrame)
    if not per_event.empty:
        assert {"Year", "Race_Num", "event_mae"}.issubset(per_event.columns)

