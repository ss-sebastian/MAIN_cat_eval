import json

from main_eval.constants import ITEM_IDS
from main_eval.response import parse_and_validate_response


def _items(scores=None, drop=None, extra=None):
    scores = scores if scores is not None else {i: (2 if i == "A1" else 1) for i in ITEM_IDS}
    items = []
    for i in ITEM_IDS:
        if drop and i in drop:
            continue
        items.append({"item_id": i, "score": scores.get(i, 0), "evidence": [], "rationale": ""})
    if extra:
        items.extend(extra)
    return items


def _json(items):
    return json.dumps({"items": items})


def test_valid_response():
    p = parse_and_validate_response(_json(_items()))
    assert p.valid
    assert [it["item_id"] for it in p.items] == ITEM_IDS


def test_zero_scores_kept():
    p = parse_and_validate_response(_json(_items(scores={i: 0 for i in ITEM_IDS})))
    assert p.valid
    assert all(v == 0 for v in p.scores.values())


def test_missing_item_invalid():
    p = parse_and_validate_response(_json(_items(drop={"A5"})))
    assert not p.valid
    assert "A5" in p.error


def test_duplicate_item_invalid():
    p = parse_and_validate_response(_json(_items(extra=[{"item_id": "A1", "score": 1, "evidence": [], "rationale": ""}])))
    assert not p.valid
    assert "duplicate" in p.error


def test_unknown_item_invalid():
    p = parse_and_validate_response(_json(_items(extra=[{"item_id": "A99", "score": 1, "evidence": [], "rationale": ""}])))
    assert not p.valid


def test_out_of_range_invalid():
    s = {i: 1 for i in ITEM_IDS}
    s["A1"] = 3
    p = parse_and_validate_response(_json(_items(scores=s)))
    assert not p.valid
    assert "A1" in p.error


def test_non_integer_invalid():
    s = {i: 1 for i in ITEM_IDS}
    s["A2"] = 1.5
    p = parse_and_validate_response(_json(_items(scores=s)))
    assert not p.valid


def test_bool_score_invalid():
    s = {i: 1 for i in ITEM_IDS}
    s["A2"] = True
    p = parse_and_validate_response(_json(_items(scores=s)))
    assert not p.valid


def test_empty_evidence_ok():
    p = parse_and_validate_response(_json(_items()))
    assert p.valid
    assert p.items[0]["evidence"] == []


def test_non_list_evidence_invalid():
    items = _items()
    items[0]["evidence"] = "not a list"
    p = parse_and_validate_response(_json(items))
    assert not p.valid


def test_markdown_fence_tolerated():
    raw = "```json\n" + _json(_items()) + "\n```"
    p = parse_and_validate_response(raw)
    assert p.valid


def test_non_json_invalid():
    p = parse_and_validate_response("this is not json")
    assert not p.valid


def test_missing_items_key_invalid():
    p = parse_and_validate_response(json.dumps({"foo": []}))
    assert not p.valid
