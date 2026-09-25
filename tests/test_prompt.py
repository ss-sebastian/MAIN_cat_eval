import json

from main_eval.errors import PromptError
from main_eval.prompt import (
    build_prompt,
    find_unreplaced_placeholders,
    format_demonstrations,
    load_prompt_template,
    template_hash,
)
from main_eval.schema import Annotation, Sample


def _template():
    return "task\n{{demonstrations}}\n{{transcript}}\n"


def test_build_prompt_replaces_placeholders():
    p = build_prompt(_template(), "DEMOS", "TEXT")
    assert "{{demonstrations}}" not in p
    assert "{{transcript}}" not in p
    assert "DEMOS" in p and "TEXT" in p


def test_unreplaced_placeholder_detected():
    t = "task\n{{demonstrations}}\n{{transcript}}\n{{MAIN_CAT_SCORING_RULES}}\n"
    try:
        build_prompt(t, "D", "T")
    except PromptError as e:
        assert "MAIN_CAT_SCORING_RULES" in str(e)
        return
    raise AssertionError("expected PromptError for unfilled rules placeholder")


def test_json_braces_not_broken():
    # Template contains a JSON example with single braces; rendering must not break.
    t = (
        'Return JSON: {"items": []}\n'
        "{{demonstrations}}\n"
        "{{transcript}}\n"
    )
    p = build_prompt(t, '{"items": [{"item_id": "A1"}]}', "T")
    assert '{"items": []}' in p
    assert '"item_id": "A1"' in p


def test_load_template_missing_placeholder():
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write("no placeholders here")
    try:
        load_prompt_template(path)
    except PromptError:
        return
    finally:
        os.remove(path)
    raise AssertionError("expected PromptError for missing placeholder")


def test_template_hash_stable():
    assert template_hash("abc") == template_hash("abc")
    assert template_hash("abc") != template_hash("abd")


def _sample():
    anns = [Annotation(item_id="A1", score=2, evidence=["e"], rationale="r")]
    return Sample(sample_id="SECRET_ID", child_id="SECRET_CHILD", story_type="Cat",
                  transcript="TRANSCRIPT", annotations=anns)


def test_format_demonstrations_excludes_metadata():
    out = format_demonstrations([_sample()])
    assert "TRANSCRIPT" in out
    assert "SECRET_ID" not in out
    assert "SECRET_CHILD" not in out
    assert '"item_id": "A1"' in out


def test_format_demonstrations_empty():
    assert format_demonstrations([]) == ""


def test_find_unreplaced_placeholders():
    assert find_unreplaced_placeholders("a {{x}} b") == ["{{x}}"]
    assert find_unreplaced_placeholders("no placeholders") == []
