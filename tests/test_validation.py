from main_eval.constants import ITEM_IDS
from main_eval.dataset import validate_dataset
from main_eval.errors import DatasetError


def _raw(sample_id, child_id, story_type="Cat", transcript="t", annotations=None, **extra):
    rec = {"sample_id": sample_id, "child_id": child_id, "story_type": story_type, "transcript": transcript}
    if annotations is not None:
        rec["annotations"] = annotations
    rec.update(extra)
    return rec


def _full_annotations():
    return [
        {"item_id": i, "score": 2 if i == "A1" else 1, "evidence": [], "rationale": ""}
        for i in ITEM_IDS
    ]


def _raises(fn, exc=DatasetError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_valid_dataset_passes():
    samples = validate_dataset(
        [_raw("s1", "c1", annotations=_full_annotations())],
        require_annotations=True,
    )
    assert samples[0].sample_id == "s1"


def test_duplicate_sample_id():
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1", annotations=_full_annotations()),
         _raw("s1", "c2", annotations=_full_annotations())],
        require_annotations=True))


def test_duplicate_child_id():
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1", annotations=_full_annotations()),
         _raw("s2", "c1", annotations=_full_annotations())],
        require_annotations=True))


def test_wrong_story_type():
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1", story_type="Dog", annotations=_full_annotations())],
        require_annotations=True))


def test_missing_required_field():
    _raises(lambda: validate_dataset(
        [{"sample_id": "s1", "child_id": "c1", "transcript": "t"}],
        require_annotations=True))


def test_score_out_of_range():
    anns = _full_annotations()
    anns[0]["score"] = 3  # A1 max is 2
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1", annotations=anns)], require_annotations=True))


def test_score_non_integer():
    for bad in (1.5, "1", True, None):
        anns = _full_annotations()
        anns[1]["score"] = bad  # A2
        _raises(lambda: validate_dataset(
            [_raw("s1", "c1", annotations=anns)], require_annotations=True))


def test_missing_annotations_when_required():
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1")], require_annotations=True))


def test_incomplete_annotations():
    anns = _full_annotations()[:-1]  # drop A16
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1", annotations=anns)], require_annotations=True))


def test_duplicate_item_id():
    anns = _full_annotations()
    anns[15]["item_id"] = "A1"  # duplicate
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1", annotations=anns)], require_annotations=True))


def test_eval_without_annotations_allowed():
    samples = validate_dataset([_raw("s1", "c1")], require_annotations=False)
    assert samples[0].annotations == []


def test_empty_transcript_rejected():
    _raises(lambda: validate_dataset(
        [_raw("s1", "c1", transcript="  ", annotations=_full_annotations())],
        require_annotations=True))
