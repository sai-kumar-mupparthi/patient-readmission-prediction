import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readmission_demo import make_encounters, run  # noqa: E402


def test_synthetic_data_is_reproducible():
    assert make_encounters(500).equals(make_encounters(500))


def test_model_beats_chance_and_is_fast():
    r = run(6000)
    assert max(r["gbm_test_roc_auc"], r["logreg_test_roc_auc"]) > 0.65
    assert r["recall_at_threshold"] > 0.6
    assert r["single_prediction_ms"] < 100
