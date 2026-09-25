"""Inter-rater agreement between model predictions and human gold scores.

"Agreement" here means agreement between the model (treated as one rater) and
the human gold annotation (treated as the second rater). All functions are pure
and dependency-free. If you later need human-human reliability, provide multiple
human raters per sample and reuse :func:`krippendorff_alpha`, which supports any
number of raters.

Coefficients implemented:
- percent agreement
- Cohen's kappa (unweighted, nominal) for two raters
- Cohen's weighted kappa (linear and quadratic weights) for ordinal scales
- Krippendorff's alpha (nominal and ordinal distance metrics), which generalizes
  to any number of raters and tolerates missing data

Conventions / interpretation notes:
- Values are in [-1, 1] for kappa and alpha; 1 = perfect agreement, 0 = chance
  agreement, negative = worse than chance. Percent agreement is in [0, 1].
- A1 (0-2) and the total score (0-17) are ordinal, so weighted kappa and ordinal
  Krippendorff's alpha are the appropriate agreement coefficients for them.
- A2-A16 are binary, so unweighted Cohen's kappa applies.
- "Chance" corrections use each rater's own marginal distribution (Cohen) or the
  coincidence marginals (Krippendorff).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from .constants import ITEM_IDS

_EPS = 1e-12


def percent_agreement(a: Sequence, b: Sequence) -> float | None:
    """Fraction of aligned positions where the two raters agree (0..1)."""
    a = list(a)
    b = list(b)
    n = len(a)
    if n == 0:
        return None
    if n != len(b):
        raise ValueError("sequences must be aligned and of equal length")
    return sum(1 for x, y in zip(a, b) if x == y) / n


def cohens_kappa(a: Sequence, b: Sequence) -> float | None:
    """Cohen's (unweighted) kappa for two raters over nominal categories."""
    a = list(a)
    b = list(b)
    n = len(a)
    if n == 0:
        return None
    if n != len(b):
        raise ValueError("sequences must be aligned and of equal length")

    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    categories = set(ca) | set(cb)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in categories)
    if 1.0 - expected < _EPS:
        # Both raters used a single, identical category: agreement is trivially perfect.
        return 1.0
    return (observed - expected) / (1.0 - expected)


def cohens_weighted_kappa(
    a: Sequence,
    b: Sequence,
    weights: str = "quadratic",
    categories: Sequence | None = None,
) -> float | None:
    """Cohen's weighted kappa for two raters over an ordinal scale.

    ``weights`` is ``"linear"`` (w = 1 - |i-j|/(k-1)) or ``"quadratic"``
    (w = 1 - (i-j)^2/(k-1)^2). ``categories`` is the ordered list of levels;
    if omitted it is derived as the sorted unique values.
    """
    a = list(a)
    b = list(b)
    n = len(a)
    if n == 0:
        return None
    if n != len(b):
        raise ValueError("sequences must be aligned and of equal length")

    if categories is None:
        categories = sorted(set(a) | set(b))
    cats = list(categories)
    k = len(cats)
    if k < 2:
        return 1.0 if a == b else 0.0
    idx = {v: i for i, v in enumerate(cats)}

    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1

    if weights == "linear":
        w = [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[1.0 - ((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    else:
        raise ValueError("weights must be 'linear' or 'quadratic'")

    po = sum(obs[i][j] * w[i][j] for i in range(k) for j in range(k)) / n
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row[i] * col[j] * w[i][j] for i in range(k) for j in range(k)) / (n * n)
    if 1.0 - pe < _EPS:
        return 1.0 if po >= 1.0 - _EPS else 0.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha(
    ratings: Iterable[Sequence],
    metric: str = "nominal",
    categories: Sequence | None = None,
) -> float | None:
    """Krippendorff's alpha over a set of units, each with one or more raters.

    ``ratings`` is an iterable of units; each unit is a sequence of rater values
    (``None`` entries are treated as missing). ``metric`` is ``"nominal"``
    (delta = 0 if equal else 1) or ``"ordinal"`` (delta = squared rank difference).
    """
    units = [[v for v in u if v is not None] for u in ratings]

    if categories is None:
        cat_set = set()
        for u in units:
            cat_set.update(u)
        categories = sorted(cat_set)
    cats = list(categories)
    k = len(cats)
    idx = {v: i for i, v in enumerate(cats)}

    if metric == "nominal":
        def delta_sq(i: int, j: int) -> float:
            return 0.0 if i == j else 1.0
    elif metric == "ordinal":
        def delta_sq(i: int, j: int) -> float:
            return float((i - j) ** 2)
    else:
        raise ValueError("metric must be 'nominal' or 'ordinal'")

    # Coincidence matrix (symmetric), each pair weighted by 1/(m_u - 1).
    o = [[0.0] * k for _ in range(k)]
    for u in units:
        m = len(u)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for i in range(m):
            for j in range(i + 1, m):
                ci = idx[u[i]]
                cj = idx[u[j]]
                o[ci][cj] += weight
                o[cj][ci] += weight

    n = sum(sum(row) for row in o)
    if n < 2:
        return None
    marginals = [sum(o[i]) for i in range(k)]

    observed_disagreement = sum(o[i][j] * delta_sq(i, j) for i in range(k) for j in range(k)) / n
    expected_disagreement = (
        sum(marginals[i] * marginals[j] * delta_sq(i, j) for i in range(k) for j in range(k))
        / (n * (n - 1))
    )
    if expected_disagreement < _EPS:
        return 1.0 if observed_disagreement < _EPS else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _agreement_block(rows: list[tuple[dict, dict]]) -> dict:
    """Compute per-item and total-score agreement for one set of (record, gold) rows."""
    items: dict[str, dict] = {}
    for iid in ITEM_IDS:
        pred = [r["scores"][iid] for r, _ in rows]
        gold = [g[iid] for r, g in rows]
        entry = {
            "percent_agreement": _round(percent_agreement(pred, gold)),
            "cohens_kappa": _round(cohens_kappa(pred, gold)),
        }
        if iid == "A1":  # ordinal 0-2
            entry["weighted_kappa_linear"] = _round(cohens_weighted_kappa(pred, gold, "linear"))
            entry["weighted_kappa_quadratic"] = _round(cohens_weighted_kappa(pred, gold, "quadratic"))
            entry["krippendorff_alpha_ordinal"] = _round(
                krippendorff_alpha([[p, g] for p, g in zip(pred, gold)], "ordinal")
            )
        items[iid] = entry

    pred_total = [r["scores"]["total"] for r, _ in rows]
    gold_total = [sum(g.values()) for r, g in rows]
    units = [[p, g] for p, g in zip(pred_total, gold_total)]
    total = {
        "weighted_kappa_linear": _round(cohens_weighted_kappa(pred_total, gold_total, "linear")),
        "weighted_kappa_quadratic": _round(cohens_weighted_kappa(pred_total, gold_total, "quadratic")),
        "krippendorff_alpha_nominal": _round(krippendorff_alpha(units, "nominal")),
        "krippendorff_alpha_ordinal": _round(krippendorff_alpha(units, "ordinal")),
    }

    pa_values = [v["percent_agreement"] for v in items.values() if v["percent_agreement"] is not None]
    kappa_values = [v["cohens_kappa"] for v in items.values() if v["cohens_kappa"] is not None]
    return {
        "n": len(rows),
        "items": items,
        "total": total,
        "macro_percent_agreement": _round(sum(pa_values) / len(pa_values)) if pa_values else None,
        "macro_cohens_kappa": _round(sum(kappa_values) / len(kappa_values)) if kappa_values else None,
    }


def compute_agreement(records: Sequence[dict], human_by_sample: dict[str, dict[str, int]]) -> dict:
    """Compute model-vs-human agreement from saved records, grouped like metrics.

    Only records with ``parse_status == "valid"`` and a labeled sample contribute.
    Repeats are treated as separate (record, gold) pairs, consistent with the
    existing ``compute_metrics`` convention.
    """
    valid = [r for r in records if r["parse_status"] == "valid" and r["sample_id"] in human_by_sample]

    def block(rs: list[dict]) -> dict | None:
        rows = [(r, human_by_sample[r["sample_id"]]) for r in rs]
        return _agreement_block(rows) if rows else None

    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_repeat: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_condition[f"shots={r['n_shots']},seed={r['sample_seed']}"].append(r)
        by_repeat[str(r["repeat_id"])].append(r)

    return {
        "n_valid": len(valid),
        "note": (
            "agreement between model predictions and human gold; values in [-1,1] (1=perfect, "
            "0=chance); A1 and total are ordinal (weighted kappa / ordinal alpha), A2-A16 binary "
            "(unweighted kappa)"
        ),
        "overall": block(valid),
        "by_condition": {k: block(v) for k, v in sorted(by_condition.items())},
        "by_repeat": {k: block(v) for k, v in sorted(by_repeat.items(), key=lambda kv: int(kv[0]))},
    }
