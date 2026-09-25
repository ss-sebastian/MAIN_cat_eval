"""Derive programmatic subtotals and total from validated A1-A16 scores."""

from __future__ import annotations

from .constants import EPISODES, ITEM_IDS


def compute_scores(scores: dict[str, int]) -> dict[str, int]:
    """Compute derived subtotals and total.

    ``scores`` must be a validated {item_id: int} map covering all of A1-A16.
    Returns a flat dict with each item, the four subtotals, and the total
    (0-17). Totals are program-derived, never produced by the model.
    """
    missing = [iid for iid in ITEM_IDS if iid not in scores]
    if missing:
        raise ValueError("compute_scores requires all A1-A16 items; missing: " + ", ".join(missing))

    out = {iid: scores[iid] for iid in ITEM_IDS}
    scene = sum(scores[iid] for iid in EPISODES["scene"])
    ep1 = sum(scores[iid] for iid in EPISODES["episode1"])
    ep2 = sum(scores[iid] for iid in EPISODES["episode2"])
    ep3 = sum(scores[iid] for iid in EPISODES["episode3"])
    out["scene"] = scene
    out["episode1"] = ep1
    out["episode2"] = ep2
    out["episode3"] = ep3
    out["total"] = scene + ep1 + ep2 + ep3
    return out
