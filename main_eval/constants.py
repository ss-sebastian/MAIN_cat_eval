"""Domain constants for the MAIN Cat story scoring rubric.

These encode *only* the structural facts given in the project specification
(item IDs, score ranges, episode grouping). No boundary-scoring content is
invented here; that belongs to the user-supplied official manual, which is
loaded from a prompt template file at run time.
"""

from __future__ import annotations

# The only story type handled by this project.
STORY_TYPE = "Cat"

# Ordered list of MAIN item IDs (A1 is the story-structure / "setting" score).
ITEM_IDS = [
    "A1", "A2", "A3", "A4", "A5", "A6",
    "A7", "A8", "A9", "A10", "A11",
    "A12", "A13", "A14", "A15", "A16",
]

# Allowed integer score values per item.
ITEM_SCORES = {
    "A1": (0, 1, 2),
    "A2": (0, 1), "A3": (0, 1), "A4": (0, 1), "A5": (0, 1), "A6": (0, 1),
    "A7": (0, 1), "A8": (0, 1), "A9": (0, 1), "A10": (0, 1), "A11": (0, 1),
    "A12": (0, 1), "A13": (0, 1), "A14": (0, 1), "A15": (0, 1), "A16": (0, 1),
}

# Episode grouping used to derive the programmatic subtotals.
EPISODES = {
    "scene": ["A1"],
    "episode1": ["A2", "A3", "A4", "A5", "A6"],
    "episode2": ["A7", "A8", "A9", "A10", "A11"],
    "episode3": ["A12", "A13", "A14", "A15", "A16"],
}

# Placeholders the prompt template must contain and that build_prompt replaces.
TEMPLATE_PLACEHOLDERS = ("{{demonstrations}}", "{{transcript}}")

# Explicit field whitelist: only these fields may ever reach the model.
# (sample_id / child_id / story_type are management metadata and are excluded.)
VISIBLE_FIELDS = ("transcript", "annotations")
