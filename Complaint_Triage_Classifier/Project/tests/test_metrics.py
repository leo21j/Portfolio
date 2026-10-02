import numpy as np

from complaint_triage.modeling.metrics import compute_basic_metrics


def test_compute_basic_metrics_returns_macro_f1():
    metrics = compute_basic_metrics(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["f1_macro"] <= 1.0
