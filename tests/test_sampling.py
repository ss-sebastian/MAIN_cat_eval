from main_eval.errors import SamplingError
from main_eval.sampling import select_demonstrations
from main_eval.schema import Sample


def make(sample_id, child_id):
    return Sample(sample_id=sample_id, child_id=child_id, story_type="Cat", transcript="t", annotations=[])


def _pool(n, reversed_ids=True):
    # Intentionally unsorted order to exercise the stable sort by sample_id.
    ids = [f"s{i:02d}" for i in (range(n, 0, -1) if reversed_ids else range(1, n + 1))]
    return [make(i, f"c{i}") for i in ids]


def _ids(sel):
    return [s.sample_id for s in sel]


def test_reproducible_same_seed_same_order():
    pool = _pool(10)
    assert _ids(select_demonstrations(pool, 42, 5)) == _ids(select_demonstrations(pool, 42, 5))


def test_input_order_independent():
    pool_a = [make("b", "cb"), make("a", "ca"), make("c", "cc")]
    pool_b = [make("c", "cc"), make("a", "ca"), make("b", "cb")]
    assert _ids(select_demonstrations(pool_a, 5, 3)) == _ids(select_demonstrations(pool_b, 5, 3))


def test_nesting_prefix_property():
    pool = _pool(10)
    one = _ids(select_demonstrations(pool, 7, 1))
    two = _ids(select_demonstrations(pool, 7, 2))
    five = _ids(select_demonstrations(pool, 7, 5))
    assert two[:1] == one
    assert five[:2] == two


def test_zero_shots_empty():
    assert select_demonstrations(_pool(5), 42, 0) == []


def test_k_exceeds_pool_raises():
    try:
        select_demonstrations(_pool(3), 42, 4)
    except SamplingError:
        return
    raise AssertionError("expected SamplingError for k > pool size")


def test_negative_k_raises():
    try:
        select_demonstrations(_pool(3), 42, -1)
    except SamplingError:
        return
    raise AssertionError("expected SamplingError for negative k")


def test_different_seed_gives_different_order():
    pool = _pool(10)
    assert _ids(select_demonstrations(pool, 1, 5)) != _ids(select_demonstrations(pool, 2, 5))
