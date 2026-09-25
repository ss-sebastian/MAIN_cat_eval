from main_eval.dataset import validate_child_separation
from main_eval.errors import ChildSeparationError
from main_eval.schema import Sample


def make(sample_id, child_id):
    return Sample(sample_id=sample_id, child_id=child_id, story_type="Cat", transcript="t", annotations=[])


def test_no_overlap_ok():
    examples = [make("s1", "c1"), make("s2", "c2")]
    evals = [make("e1", "c3")]
    validate_child_separation(examples, evals)  # should not raise


def test_overlap_raises():
    examples = [make("s1", "c1"), make("s2", "c2")]
    evals = [make("e1", "c2")]
    try:
        validate_child_separation(examples, evals)
    except ChildSeparationError as e:
        assert "c2" in str(e)
        return
    raise AssertionError("expected ChildSeparationError")
