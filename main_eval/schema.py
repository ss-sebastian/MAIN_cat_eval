"""Data model for samples and annotations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Annotation:
    """A single human- or model-produced item annotation."""

    item_id: str
    score: int
    evidence: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "evidence": list(self.evidence),
            "rationale": self.rationale,
        }


@dataclass
class Sample:
    """One child's complete Cat story narrative."""

    sample_id: str
    child_id: str
    story_type: str
    transcript: str
    annotations: list[Annotation] = field(default_factory=list)

    def annotation_map(self) -> dict[str, int]:
        """item_id -> score for quick lookup."""
        return {a.item_id: a.score for a in self.annotations}
