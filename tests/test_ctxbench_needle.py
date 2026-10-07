"""Tests for needle planting and scoring.

The scorer is the part that can quietly lie. Every case below is a way a retrieval probe
can produce a number that looks like a finding and is not: a reply that is nearly right,
a reply wrapped in a fence, a reply that reproduces the function instead of the
signature, or a needle that lands in a file the model was never shown.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from coding_agent_evaluations.ctxbench.filler import build_filler, encoding_for
from coding_agent_evaluations.ctxbench.needle import (
    make_needle,
    plant,
    score,
    signature_needle,
)

ENCODING = encoding_for("o200k_base")


@pytest.fixture(scope="module")
def corpus() -> str:
    return build_filler(16_000, ENCODING).text


# ---------------------------------------------------------------- planting


def test_plant_reports_the_module_it_landed_in(corpus: str) -> None:
    """A depth with no file behind it cannot be reproduced or audited."""
    needle = make_needle(random.Random(0), module="", depth=0.5)
    placement = plant(corpus, needle)
    assert placement.module.endswith(".py")
    assert placement.module in placement.text
    assert placement.module_count > 1
    assert 0.0 <= placement.actual_depth <= 1.0


@pytest.mark.parametrize("depth", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_plant_lands_at_the_requested_depth(corpus: str, depth: float) -> None:
    needle = make_needle(random.Random(1), module="", depth=depth)
    placement = plant(corpus, needle)
    assert abs(placement.actual_depth - depth) < 0.1


def test_plant_keeps_the_corpus_valid(corpus: str) -> None:
    """The planted file has to parse, or the model is reading broken input."""
    import ast

    needle = make_needle(random.Random(2), module="", depth=0.5)
    placement = plant(corpus, needle)
    ast.parse(placement.text)


@pytest.mark.parametrize("depth", [0.0, 1.0])
def test_plant_at_the_extremes_still_lands_in_the_corpus(corpus: str, depth: float) -> None:
    """Depth 0 and 1 are the ends of the module list, not off the end of the text: a
    needle the model was never shown would score as a context-length failure."""
    from coding_agent_evaluations.ctxbench.needle import needle_block

    needle = make_needle(random.Random(3), module="", depth=depth)
    placement = plant(corpus, needle)
    assert placement.text.count(needle_block(replace(needle, module=placement.module))) == 1
    assert placement.module_index in {0, placement.module_count - 1}
    assert placement.text.startswith("# module_") or placement.text.startswith("module_")


# ------------------------------------------------------------ literal scoring


def test_exact_answer_scores() -> None:
    needle = make_needle(random.Random(4), module="m.py", depth=0.5)
    assert score(needle, needle.answer)
    assert score(needle, f"  {needle.answer}  ")
    assert score(needle, f"`{needle.answer}`")


def test_wrong_answer_does_not_score() -> None:
    needle = make_needle(random.Random(5), module="m.py", depth=0.5)
    assert not score(needle, "sk_test_00000000000")
    assert not score(needle, "I could not find that constant.")
    assert not score(needle, "")


def test_near_miss_does_not_score() -> None:
    """A truncated or extended secret is a different string, and counting it would make
    the pass rate a measure of how forgiving the grader is."""
    needle = make_needle(random.Random(6), module="m.py", depth=0.5)
    assert not score(needle, needle.answer[:-1])
    assert not score(needle, needle.answer + "x")
    assert not score(needle, needle.answer.replace("_", "-"))
    # Duplicated output has no token boundary around the answer, so it is not a hit.
    assert not score(needle, needle.answer + needle.answer)


@pytest.mark.parametrize(
    "wrap",
    [
        "the value is {answer}",
        "{answer}",
        "`{answer}`",
        "```\n{answer}\n```",
        "```python\n{answer}\n```",
        '"{answer}"',
        "{answer}.",
        "SECRET_API_KEY = {answer}",
    ],
)
def test_packaging_around_the_value_still_scores(wrap: str) -> None:
    """Exactness belongs on the value, not on the sentence around it. A model that writes
    `The value is X.` has retrieved the needle; scoring that as a miss measures packaging."""
    needle = make_needle(random.Random(16), module="m.py", depth=0.5)
    assert score(needle, wrap.format(answer=needle.answer))


def test_a_different_value_does_not_score_inside_a_sentence() -> None:
    needle = make_needle(random.Random(17), module="m.py", depth=0.5)
    assert not score(needle, "the value is sk_test_99999999999")


def test_literal_kind_is_not_a_signature() -> None:
    needle = make_needle(random.Random(7), module="m.py", depth=0.5)
    assert needle.kind in {
        "secret_constant",
        "magic_string",
        "function_contract",
    }


# ----------------------------------------------------------- signature scoring


def test_exact_signature_scores() -> None:
    needle = signature_needle(random.Random(8), module="m.py", depth=0.5)
    assert score(needle, needle.answer)


def test_signature_in_a_fence_scores() -> None:
    """Models wrap answers in fences constantly; a fence is presentation, not a miss."""
    needle = signature_needle(random.Random(9), module="m.py", depth=0.5)
    assert score(needle, f"```python\n{needle.answer}\n    return 0\n```")


def test_signature_reproduced_without_annotations_scores() -> None:
    """Recovering the function means recovering its name, parameters and order. Dropping
    the annotation is a formatting difference, not a retrieval failure."""
    needle = signature_needle(random.Random(10), module="m.py", depth=0.5)
    stripped = needle.answer.split(" -> ")[0] + ":"
    assert score(needle, stripped)


def test_signature_with_wrong_parameters_does_not_score() -> None:
    needle = signature_needle(random.Random(11), module="m.py", depth=0.5)
    wrong = needle.answer.replace("subtotal", "total")
    assert not score(needle, wrong)


def test_signature_with_extra_parameter_does_not_score() -> None:
    needle = signature_needle(random.Random(12), module="m.py", depth=0.5)
    # Anchored on the closing paren of the signature, which is where a hallucinated extra
    # parameter would go.
    head, _, tail = needle.answer.partition(")")
    wrong = f"{head}, currency: str){tail}"
    assert wrong != needle.answer
    assert not score(needle, wrong)


def test_unparsable_signature_reply_does_not_score() -> None:
    """Counting an unparsable reply as a near-miss would grade the grader, not the model."""
    needle = signature_needle(random.Random(13), module="m.py", depth=0.5)
    assert not score(needle, "I think it takes a subtotal and a rate.")
    assert not score(needle, "")


def test_wrong_function_name_does_not_score() -> None:
    needle = signature_needle(random.Random(14), module="m.py", depth=0.5)
    wrong = needle.answer.replace(needle.name, "compute_tax")
    assert not score(needle, wrong)