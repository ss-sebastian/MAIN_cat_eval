from main_eval.agreement import (
    cohens_kappa,
    cohens_weighted_kappa,
    compute_agreement,
    krippendorff_alpha,
    percent_agreement,
)
from main_eval.constants import ITEM_IDS
from main_eval.scoring import compute_scores


def test_percent_agreement():
    assert percent_agreement([0, 1, 1], [0, 1, 0]) == 2 / 3
    assert percent_agreement([0, 1], [0, 1]) == 1.0
    assert percent_agreement([], []) is None


def test_cohens_kappa_known_value():
    a = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    # observed = 0.8, expected = 0.3*0.3 + 0.7*0.7 = 0.58 -> kappa = 0.22/0.42
    assert abs(cohens_kappa(a, b) - 0.523810) < 1e-5


def test_cohens_kappa_perfect():
    assert cohens_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0


def test_cohens_kappa_single_category():
    # Both raters constant on the same category -> trivially perfect.
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_weighted_kappa_perfect():
    assert cohens_weighted_kappa([0, 1, 2], [0, 1, 2], "linear") == 1.0
    assert cohens_weighted_kappa([0, 1, 2], [0, 1, 2], "quadratic") == 1.0


def test_weighted_kappa_quadratic_reversed():
    # [0,1,2] vs [2,1,0] -> quadratic weighted kappa == -1.0
    assert abs(cohens_weighted_kappa([0, 1, 2], [2, 1, 0], "quadratic") - (-1.0)) < 1e-9


def test_weighted_kappa_invalid_weights():
    try:
        cohens_weighted_kappa([0, 1], [0, 1], "bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_krippendorff_perfect_agreement():
    assert abs(krippendorff_alpha([[0, 0], [1, 1], [1, 1]], "nominal") - 1.0) < 1e-9


def test_krippendorff_perfect_disagreement():
    # 2 raters, 4 units, always disagree -> alpha == -0.75
    ratings = [[0, 1], [1, 0], [0, 1], [1, 0]]
    assert abs(krippendorff_alpha(ratings, "nominal") - (-0.75)) < 1e-9


def test_krippendorff_ordinal_perfect():
    assert abs(krippendorff_alpha([[0, 0], [2, 2], [1, 1]], "ordinal") - 1.0) < 1e-9


def test_krippendorff_missing_data():
    # One unit missing a rater is skipped; remaining units still agree.
    ratings = [[0, 0], [1, 1], [1, None]]
    assert abs(krippendorff_alpha(ratings, "nominal") - 1.0) < 1e-9


def _record(sample_id, repeat_id, n_shots, seed, pred_scores, parse_status="valid"):
    items = [{"item_id": i, "score": pred_scores[i], "evidence": [], "rationale": ""} for i in ITEM_IDS]
    scores = compute_scores(pred_scores) if parse_status == "valid" else {}
    return {
        "sample_id": sample_id, "repeat_id": repeat_id, "n_shots": n_shots,
        "sample_seed": seed, "parse_status": parse_status, "items": items, "scores": scores,
    }


def _gold(a1=2, others=1):
    return {i: (a1 if i == "A1" else others) for i in ITEM_IDS}


def test_compute_agreement_perfect():
    gold = _gold()
    human = {"s1": gold}
    recs = [_record("s1", 1, 1, 0, gold)]
    a = compute_agreement(recs, human)
    assert a["n_valid"] == 1
    assert a["overall"]["macro_cohens_kappa"] == 1.0
    assert a["overall"]["total"]["weighted_kappa_quadratic"] == 1.0
    assert a["overall"]["items"]["A1"]["weighted_kappa_quadratic"] == 1.0


def test_compute_agreement_excludes_invalid_and_unlabeled():
    gold = _gold()
    human = {"s1": gold}
    recs = [
        _record("s1", 1, 1, 0, gold),
        _record("s2", 1, 1, 0, gold),  # unlabeled sample
        _record("s1", 2, 1, 0, gold, parse_status="invalid"),
    ]
    a = compute_agreement(recs, human)
    assert a["n_valid"] == 1


def test_compute_agreement_groups():
    gold = _gold()
    human = {"s1": gold, "s2": gold}
    recs = [
        _record("s1", 1, 1, 0, gold),
        _record("s2", 1, 1, 0, gold),
        _record("s1", 1, 2, 7, gold),
    ]
    a = compute_agreement(recs, human)
    assert "shots=1,seed=0" in a["by_condition"]
    assert "shots=2,seed=7" in a["by_condition"]
    assert "1" in a["by_repeat"]
