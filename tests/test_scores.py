from main_eval.constants import ITEM_IDS
from main_eval.scoring import compute_scores


def _all(score):
    return {i: score for i in ITEM_IDS}


def test_all_ones_subtotals():
    out = compute_scores(_all(1))
    assert out["scene"] == 1
    assert out["episode1"] == 5
    assert out["episode2"] == 5
    assert out["episode3"] == 5
    assert out["total"] == 16


def test_max_total_17():
    s = _all(1)
    s["A1"] = 2
    out = compute_scores(s)
    assert out["total"] == 17
    assert out["scene"] == 2


def test_zero_total():
    out = compute_scores(_all(0))
    assert out["total"] == 0


def test_episode_boundaries():
    s = _all(0)
    s["A2"] = 1
    s["A6"] = 1
    s["A7"] = 1
    s["A16"] = 1
    out = compute_scores(s)
    assert out["episode1"] == 2  # A2, A6
    assert out["episode2"] == 1  # A7
    assert out["episode3"] == 1  # A16
    assert out["total"] == 4


def test_missing_item_raises():
    s = _all(1)
    del s["A3"]
    try:
        compute_scores(s)
    except ValueError as e:
        assert "A3" in str(e)
        return
    raise AssertionError("expected ValueError")
