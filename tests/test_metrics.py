from main_eval.constants import ITEM_IDS
from main_eval.metrics import compute_metrics, summarize_repeat_stability
from main_eval.scoring import compute_scores


def _record(sample_id, repeat_id, n_shots, seed, pred_scores, parse_status="valid"):
    items = [
        {"item_id": i, "score": pred_scores[i], "evidence": [], "rationale": ""}
        for i in ITEM_IDS
    ]
    scores = compute_scores(pred_scores) if parse_status == "valid" else {}
    return {
        "sample_id": sample_id, "repeat_id": repeat_id, "n_shots": n_shots,
        "sample_seed": seed, "parse_status": parse_status, "items": items, "scores": scores,
    }


def _gold(a1=2, others=1):
    return {i: (a1 if i == "A1" else others) for i in ITEM_IDS}


def test_perfect_accuracy_and_zero_mae():
    gold = _gold()
    human = {"s1": gold}
    recs = [_record("s1", 1, 1, 0, gold)]
    m = compute_metrics(recs, human)
    assert m["overall"]["macro_accuracy"] == 1.0
    assert m["overall"]["total_mae"] == 0.0
    assert m["overall"]["total_mean_signed_error"] == 0.0
    assert m["invalid_outputs"]["count"] == 0


def test_a1_confusion_matrix():
    gold = _gold(a1=2)
    pred = _gold(a1=0)  # model predicted A1=0
    human = {"s1": gold}
    m = compute_metrics([_record("s1", 1, 1, 0, pred)], human)
    conf = m["overall"]["A1_confusion"]
    assert conf[2][0] == 1  # gold=2, pred=0


def test_binary_precision_recall():
    gold = _gold(others=1)
    pred = _gold(others=0)  # predict all negatives for A2-A16
    human = {"s1": gold}
    m = compute_metrics([_record("s1", 1, 1, 0, pred)], human)
    b = m["overall"]["binary_metrics"]["A2"]
    assert b["precision"] == 0.0  # no true positives
    assert b["recall"] == 0.0
    assert b["support"] == 1  # one positive in gold


def test_mae_and_signed_error():
    gold = _gold(a1=2)  # total 17
    pred = _gold(a1=0, others=0)  # total 0
    human = {"s1": gold}
    m = compute_metrics([_record("s1", 1, 1, 0, pred)], human)
    assert m["overall"]["total_mae"] == 17.0
    assert m["overall"]["total_mean_signed_error"] == -17.0


def test_invalid_outputs_reported_separately():
    gold = _gold()
    human = {"s1": gold, "s2": gold}
    recs = [
        _record("s1", 1, 1, 0, gold),
        _record("s2", 1, 1, 0, gold, parse_status="invalid"),
    ]
    m = compute_metrics(recs, human)
    assert m["invalid_outputs"]["count"] == 1
    assert m["invalid_outputs"]["labeled_total"] == 2
    assert m["invalid_outputs"]["rate"] == 0.5
    assert m["overall"]["n"] == 1  # only the valid record


def test_stability_identical_repeats():
    gold = _gold()
    recs = [_record("s1", 1, 1, 0, gold), _record("s1", 2, 1, 0, gold)]
    s = summarize_repeat_stability(recs)
    assert s["estimable"] is True
    assert s["item_agreement_rate"] == 1.0
    assert s["pairwise_disagreement_rate"] == 0.0
    assert s["total_range"]["max"] == 0


def test_stability_disagreement_detected():
    recs = [
        _record("s1", 1, 1, 0, _gold(a1=2)),
        _record("s1", 2, 1, 0, _gold(a1=0)),
    ]
    s = summarize_repeat_stability(recs)
    assert s["estimable"] is True
    assert s["pairwise_disagreement_rate"] > 0.0
    assert s["total_range"]["max"] == 2


def test_stability_not_estimable_single_repeat():
    recs = [_record("s1", 1, 1, 0, _gold())]
    s = summarize_repeat_stability(recs)
    assert s["estimable"] is False


def test_stability_ignores_invalid_records():
    gold = _gold()
    recs = [
        _record("s1", 1, 1, 0, gold),
        _record("s1", 2, 1, 0, gold, parse_status="invalid"),
    ]
    s = summarize_repeat_stability(recs)
    assert s["estimable"] is False  # only one valid repeat


def test_rationale_wording_not_compared():
    # Same scores, different rationale -> still full agreement.
    gold = _gold()
    r1 = _record("s1", 1, 1, 0, gold)
    r2 = _record("s1", 2, 1, 0, gold)
    r1["items"][0]["rationale"] = "wording A"
    r2["items"][0]["rationale"] = "wording B"
    s = summarize_repeat_stability([r1, r2])
    assert s["item_agreement_rate"] == 1.0
