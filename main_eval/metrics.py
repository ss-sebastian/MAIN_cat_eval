"""Metric computation over saved records (pure functions).

Only records whose eval sample has human annotations contribute to accuracy /
F1 / MAE metrics. Invalid outputs are reported separately and are excluded
from the valid-sample denominators (the denominator is reported explicitly).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .constants import ITEM_IDS


def compute_metrics(records: Sequence[dict], human_by_sample: dict[str, dict[str, int]]) -> dict:
    labeled = [r for r in records if r["sample_id"] in human_by_sample]
    invalid = [r for r in labeled if r["parse_status"] != "valid"]
    valid = [r for r in labeled if r["parse_status"] == "valid"]
    n_labeled = len(labeled)

    def rows_from(rs):
        rows = []
        for r in rs:
            gold = human_by_sample[r["sample_id"]]
            pred = {it["item_id"]: it["score"] for it in r["items"]}
            pred_total = r["scores"].get("total")
            gold_total = sum(gold.values())
            rows.append((pred, gold, pred_total, gold_total))
        return rows

    def item_block(rows):
        if not rows:
            return None
        n = len(rows)
        item_acc = {}
        for iid in ITEM_IDS:
            correct = sum(1 for (p, g, _, _) in rows if p.get(iid) == g.get(iid))
            item_acc[iid] = round(correct / n, 6)

        confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # [gold][pred]
        for (p, g, _, _) in rows:
            confusion[g["A1"]][p["A1"]] += 1

        binary = {}
        for iid in ITEM_IDS[1:]:
            tp = fp = fn = 0
            for (p, g, _, _) in rows:
                a, b = p.get(iid), g.get(iid)
                if a == 1 and b == 1:
                    tp += 1
                elif a == 1 and b == 0:
                    fp += 1
                elif a == 0 and b == 1:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            binary[iid] = {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": tp + fn,
            }

        mae = sum(abs(pt - gt) for (_, _, pt, gt) in rows) / n
        signed = sum((pt - gt) for (_, _, pt, gt) in rows) / n
        macro_acc = sum(item_acc.values()) / len(item_acc)
        macro_p = sum(b["precision"] for b in binary.values()) / len(binary)
        macro_r = sum(b["recall"] for b in binary.values()) / len(binary)
        macro_f1 = sum(b["f1"] for b in binary.values()) / len(binary)
        return {
            "n": n,
            "item_accuracy": item_acc,
            "macro_accuracy": round(macro_acc, 6),
            "A1_confusion": confusion,
            "binary_metrics": binary,
            "macro_precision": round(macro_p, 6),
            "macro_recall": round(macro_r, 6),
            "macro_f1": round(macro_f1, 6),
            "total_mae": round(mae, 6),
            "total_mean_signed_error": round(signed, 6),
        }

    by_condition: dict[str, list] = defaultdict(list)
    by_repeat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_condition[f"shots={r['n_shots']},seed={r['sample_seed']}"].append(r)
        by_repeat[str(r["repeat_id"])].append(r)

    return {
        "invalid_outputs": {
            "count": len(invalid),
            "labeled_total": n_labeled,
            "rate": round(len(invalid) / n_labeled, 6) if n_labeled else None,
            "note": "valid-sample metrics use only records with parse_status=='valid' and labeled samples",
        },
        "overall": item_block(rows_from(valid)),
        "by_condition": {k: item_block(rows_from(v)) for k, v in sorted(by_condition.items())},
        "by_repeat": {
            k: item_block(rows_from(v)) for k, v in sorted(by_repeat.items(), key=lambda kv: int(kv[0]))
        },
    }



def summarize_repeat_stability(records: Sequence[dict]) -> dict:
    """Report scoring agreement across repeats of the *same* input.

    Only scores are compared; differing rationale wording with identical scores
    is NOT counted as disagreement. Samples with fewer than two valid repeats
    are excluded; if none qualify, stability is reported as not estimable.
    """
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sample[r["sample_id"]].append(r)

    groups = {
        sid: [r for r in rs if r["parse_status"] == "valid"]
        for sid, rs in by_sample.items()
    }
    multi = {sid: rs for sid, rs in groups.items() if len(rs) >= 2}
    if not multi:
        return {"estimable": False, "note": "stability not estimable (fewer than 2 valid repeats)"}

    agree_sum = 0
    n_samples = 0
    pair_disagree = 0
    pair_total = 0
    ranges = []
    for rs in multi.values():
        n_samples += 1
        for iid in ITEM_IDS:
            scores = [r["scores"][iid] for r in rs]
            if len(set(scores)) == 1:
                agree_sum += 1
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    pair_total += 1
                    if scores[i] != scores[j]:
                        pair_disagree += 1
        totals = [r["scores"]["total"] for r in rs]
        ranges.append(max(totals) - min(totals))

    denom_items = n_samples * len(ITEM_IDS)
    return {
        "estimable": True,
        "samples_with_repeats": n_samples,
        "item_agreement_rate": round(agree_sum / denom_items, 6) if denom_items else None,
        "pairwise_disagreement_rate": round(pair_disagree / pair_total, 6) if pair_total else None,
        "total_range": {
            "mean": round(sum(ranges) / len(ranges), 6),
            "min": min(ranges),
            "max": max(ranges),
        },
        "note": "only scores compared; wording differences with equal scores are not disagreement",
    }
