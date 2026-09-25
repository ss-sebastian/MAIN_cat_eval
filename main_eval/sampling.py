"""Deterministic demonstration sampling.

Guarantees (all tested):
- Examples are first sorted stably by the unique ``sample_id``.
- A local ``random.Random(sample_seed)`` shuffles that sorted order.
- The first ``k`` entries are taken (no replacement).
- Same pool + seed + k => identical examples in identical order.
- The ``k``-shot selection is a prefix of the ``k+1``-shot selection for the
  same seed (nesting property). This holds because the shuffle is computed
  once over the full list and we only take a prefix.
- ``k > len(pool)`` raises SamplingError (no sampling with replacement).
- ``k == 0`` returns an empty list.
"""

from __future__ import annotations

import random
from typing import Sequence

from .errors import SamplingError
from .schema import Sample


def select_demonstrations(
    examples: Sequence[Sample],
    sample_seed: int,
    n_shots: int,
) -> list[Sample]:
    if n_shots < 0:
        raise SamplingError(f"n_shots must be >= 0, got {n_shots}")
    if n_shots == 0:
        return []

    # Stable sort by unique sample_id (requirement 1).
    ordered = sorted(examples, key=lambda s: s.sample_id)

    if n_shots > len(ordered):
        raise SamplingError(
            f"n_shots={n_shots} exceeds available demonstration children ({len(ordered)}); "
            "sampling with replacement is not allowed"
        )

    rng = random.Random(sample_seed)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    return [ordered[i] for i in idx[:n_shots]]
