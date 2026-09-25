"""Generate clearly-labeled demo data for offline testing.

These records are HUMAN-CONSTRUCTED demonstration data, NOT real child data.
Do not treat them as real Cantonese child narratives. Scores and rationales
are authored here explicitly for illustration only.

Run:  python data/make_demo_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

ITEM_IDS = [
    "A1", "A2", "A3", "A4", "A5", "A6",
    "A7", "A8", "A9", "A10", "A11",
    "A12", "A13", "A14", "A15", "A16",
]


def anns(scores: dict[str, int], evidence: dict[str, list[str]] | None = None,
         rationale: dict[str, str] | None = None) -> list[dict]:
    evidence = evidence or {}
    rationale = rationale or {}
    return [
        {
            "item_id": i,
            "score": scores[i],
            "evidence": evidence.get(i, []),
            "rationale": rationale.get(i, ""),
        }
        for i in ITEM_IDS
    ]


def sample(sample_id, child_id, transcript, scores, evidence=None, rationale=None):
    return {
        "sample_id": sample_id,
        "child_id": child_id,
        "story_type": "Cat",
        "transcript": transcript,
        "annotations": anns(scores, evidence, rationale),
    }


# --- Demonstration pool (6 children, so 5-shot is possible) ---
EXAMPLES = [
    sample("ex_01", "demo_child_01",
           "有一隻貓見到隻雀仔喺樹上面。佢想捉隻雀仔。佢跳上去捉。但係捉唔到。貓好嬲。",
           {"A1": 2, "A2": 1, "A3": 1, "A4": 1, "A5": 0, "A6": 1,
            "A7": 0, "A8": 0, "A9": 0, "A10": 0, "A11": 0,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0},
           evidence={"A2": ["見到隻雀仔"]},
           rationale={"A1": "setting and characters present", "A2": "initiating event: saw the bird",
                      "A3": "goal: wants to catch the bird", "A4": "attempt: jumped up",
                      "A5": "outcome: did not catch it", "A6": "reaction: angry"}),
    sample("ex_02", "demo_child_02",
           "隻貓想捉雀仔。佢慢慢行過去。隻狗過嚟追貓。貓走咗去。雀仔飛走咗。",
           {"A1": 1, "A2": 0, "A3": 1, "A4": 1, "A5": 1, "A6": 0,
            "A7": 1, "A8": 1, "A9": 1, "A10": 1, "A11": 1,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0},
           rationale={"A3": "goal: wants to catch the bird", "A4": "attempt: walked over",
                      "A5": "outcome: bird flew away", "A7": "initiating event: dog chases cat",
                      "A10": "outcome: cat ran away"}),
    sample("ex_03", "demo_child_03",
           "有一日，貓見到隻雀仔。貓想捉佢。貓爬樹。狗追貓。貓好驚，跑走咗。",
           {"A1": 2, "A2": 1, "A3": 1, "A4": 1, "A5": 0, "A6": 0,
            "A7": 1, "A8": 1, "A9": 1, "A10": 1, "A11": 1,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0},
           rationale={"A1": "setting ('one day') and characters present", "A2": "saw the bird",
                      "A3": "wants to catch it", "A4": "climbed the tree", "A7": "dog chases cat",
                      "A11": "scared"}),
    sample("ex_04", "demo_child_04",
           "貓見到雀仔。佢想捉。佢跳。雀仔飛咗上天。貓望住，好失望。",
           {"A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "A6": 1,
            "A7": 0, "A8": 0, "A9": 0, "A10": 0, "A11": 0,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0},
           rationale={"A2": "saw the bird", "A3": "wants to catch", "A4": "jumped",
                      "A5": "bird flew up into the sky", "A6": "disappointed"}),
    sample("ex_05", "demo_child_05",
           "貓想食雀仔。佢慢慢行。狗出嚟，好嬲。貓跑走。",
           {"A1": 0, "A2": 0, "A3": 1, "A4": 1, "A5": 0, "A6": 0,
            "A7": 1, "A8": 0, "A9": 1, "A10": 1, "A11": 1,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0},
           rationale={"A3": "wants to eat the bird", "A4": "walked slowly",
                      "A7": "dog appeared, angry", "A11": "ran away"}),
    sample("ex_06", "demo_child_06",
           "貓見到雀仔，雀仔見到貓好驚。貓想捉雀仔，跳上嚟。雀仔飛走咗。貓好開心咁追。",
           {"A1": 2, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "A6": 1,
            "A7": 0, "A8": 0, "A9": 0, "A10": 0, "A11": 0,
            "A12": 1, "A13": 0, "A14": 0, "A15": 1, "A16": 0},
           rationale={"A1": "characters and situation present", "A2": "saw the bird",
                      "A3": "wants to catch", "A4": "jumped up", "A5": "bird flew away",
                      "A12": "bird was scared", "A15": "flew away"}),
    sample("ex_07", "demo_child_07",
           "貓見到雀仔喺地下。貓行埋去。雀仔飛走。狗追貓。貓爬上樹。",
           {"A1": 2, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "A6": 0,
            "A7": 1, "A8": 0, "A9": 1, "A10": 1, "A11": 0,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0}),
    sample("ex_08", "demo_child_08",
           "有一日貓想捉雀仔。貓慢慢行。雀仔見到貓就飛走。貓好嬲。",
           {"A1": 1, "A2": 0, "A3": 1, "A4": 1, "A5": 1, "A6": 1,
            "A7": 0, "A8": 0, "A9": 0, "A10": 0, "A11": 0,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0}),
    sample("ex_09", "demo_child_09",
           "貓見到雀仔。佢想捉佢食。佢跳過去。捉到咗。貓好開心。",
           {"A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "A6": 1,
            "A7": 0, "A8": 0, "A9": 0, "A10": 0, "A11": 0,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0}),
    sample("ex_10", "demo_child_10",
           "貓同狗都想捉雀仔。貓跑得快啲。狗唔開心。雀仔飛走咗。",
           {"A1": 1, "A2": 0, "A3": 1, "A4": 1, "A5": 1, "A6": 0,
            "A7": 1, "A8": 1, "A9": 1, "A10": 1, "A11": 1,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0}),
]

# --- Eval set (2 children, with gold annotations for metric computation) ---
EVAL = [
    sample("ev_01", "demo_eval_child_01",
           "有一隻貓同埋一隻雀仔。貓想捉雀仔。貓爬上去。雀仔飛走咗。貓好嬲。",
           {"A1": 2, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "A6": 1,
            "A7": 0, "A8": 0, "A9": 0, "A10": 0, "A11": 0,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0}),
    sample("ev_02", "demo_eval_child_02",
           "貓想捉雀仔。佢慢慢行。狗追佢。貓走咗。",
           {"A1": 0, "A2": 0, "A3": 1, "A4": 1, "A5": 0, "A6": 0,
            "A7": 1, "A8": 0, "A9": 1, "A10": 1, "A11": 0,
            "A12": 0, "A13": 0, "A14": 0, "A15": 0, "A16": 0}),
]


def main() -> None:
    here = Path(__file__).resolve().parent
    ex_path = here / "examples.jsonl"
    ev_path = here / "eval.jsonl"
    with ex_path.open("w", encoding="utf-8") as f:
        for rec in EXAMPLES:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with ev_path.open("w", encoding="utf-8") as f:
        for rec in EVAL:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(EXAMPLES)} examples -> {ex_path}")
    print(f"wrote {len(EVAL)} eval samples -> {ev_path}")


if __name__ == "__main__":
    main()
