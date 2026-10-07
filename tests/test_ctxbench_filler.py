"""Tests for the filler generator and its exact-token guarantee."""

from __future__ import annotations

import ast

import pytest

from coding_agent_evaluations.ctxbench.filler import (
    build_filler,
    count_tokens,
    encoding_for,
)

ENCODING = encoding_for("o200k_base")

SIZES = (200, 1_000, 4_000, 16_000, 32_000, 64_000, 128_000)


@pytest.mark.parametrize("size", SIZES)
def test_filler_hits_the_exact_token_target(size: int) -> None:
    """The whole point: a row labelled N is a measurement of N tokens."""
    corpus = build_filler(size, ENCODING)
    assert corpus.tokens == size
    assert count_tokens(corpus.text, ENCODING) == size


@pytest.mark.parametrize("size", SIZES)
def test_filler_is_valid_python(size: int) -> None:
    """Invalid filler would make a context-size row a measurement of broken input."""
    corpus = build_filler(size, ENCODING)
    ast.parse(corpus.text)
    # Every module must also be valid on its own, so a needle planted at a module
    # boundary lands in a file that parses.
    for block in corpus.text.split("module_")[1:]:
        ast.parse("module_" + block)


def test_filler_is_reproducible_and_seed_sensitive() -> None:
    """Same seed, same corpus: a run has to be reproducible. Different seeds, different
    filler: repeated identical filler is pattern-matchable without reading."""
    assert build_filler(16_000, ENCODING, seed=1).text == build_filler(16_000, ENCODING, seed=1).text
    assert build_filler(16_000, ENCODING, seed=1).text != build_filler(16_000, ENCODING, seed=2).text


def test_filler_grows_with_the_target() -> None:
    small = build_filler(16_000, ENCODING)
    large = build_filler(64_000, ENCODING)
    assert len(large.text) > len(small.text)
    assert large.tokens == 4 * small.tokens


def test_non_positive_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_filler(0, ENCODING)