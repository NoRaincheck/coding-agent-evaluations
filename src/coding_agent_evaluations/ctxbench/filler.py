"""Filler corpora that hit a token target exactly.

Padding a prompt to "about 16k" is not a measurement: if the real prompt is 14,300
tokens, the row labelled 16k is a 14,300-token measurement with a misleading name, and
the TTFT curve it produces is the curve for a smaller context than the one reported.

So this module builds a syntactically valid Python corpus that encodes to *exactly* a
requested token count under the configured tokenizer. Three properties matter:

* **Exactness.** The corpus is measured with the same tokenizer that counts the prompt,
  and adjusted until the encoded length equals the target.
* **Validity.** Every generated module compiles. A haystack of unparseable filler would
  let a model fail for the wrong reason, and an AST-scored needle would be unreadable.
* **Unpredictability to the model.** The filler is parameterised over a seeded PRNG with
  varied identifiers, docstrings and arithmetic, so it is not one repeated line that a
  long-context model can pattern-match instead of reading.

Adjusting to the token is done with a trailing comment rather than more code, because a
comment cannot change the module's behaviour: if the last few tokens of the haystack are
padding, the model is never asked to reason about them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import tiktoken

#: Padding units. Each entry is a self-contained block of one or more lines, emitted whole.
#: Blocks rather than single lines because a block can contain an `if` header, and a
#: corpus truncated between an `if` and its body would not parse -- which would make a
#: context-size row a measurement of invalid input rather than of the model.
_UNIT_TEMPLATES = (
    ("    accumulator_{n} = (accumulator_{n} * {a} + {b}) % {m}",),
    ("    scaled_{n} = round(accumulator_{n} / {d}, {p})",),
    ("    if scaled_{n} > {t}:", "        scaled_{n} = scaled_{n} - {t}"),
    ("    registry_{n}['{key}'] = scaled_{n}",),
    ("    total_{n} = sum(registry_{n}.values()) % {m}",),
    (
        "    if total_{n} == 0:",
        "        registry_{n}['{key}'] = accumulator_{n}",
        "        total_{n} = accumulator_{n} or 1",
    ),
)


@dataclass(frozen=True)
class FillerCorpus:
    """A validated filler codebase of a known encoded size."""

    text: str
    tokens: int
    modules: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "modules": len(self.modules)}


def encoding_for(name: str) -> tiktoken.Encoding:
    """Resolve a tokenizer by name, offline.

    Raises rather than silently falling back: a wrong tokenizer turns every context size
    in the report into a wrong number, and a wrong number here is indistinguishable from a
    real context-scaling finding.
    """
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def _module_text(index: int, lines: int, rng: random.Random) -> str:
    """One syntactically valid Python module with ``lines`` statements of filler."""
    name = f"module_{index:04d}"
    body = [
        '"""Generated filler module.',
        "",
        f"Module {index} of a synthetic corpus. Present to occupy context; carries no",
        "behaviour that any caller depends on.",
        '"""',
        "",
        "def build_registry() -> dict:",
        f'    """Return a registry sized for module {index}."""',
        f"    registry = {{'module': {index}, 'seed': {rng.randrange(10**6)}}}",
    ]
    for unit in range(lines):
        for template in _UNIT_TEMPLATES[unit % len(_UNIT_TEMPLATES)]:
            body.append(
                template.format(
                    n=unit,
                    a=rng.randrange(11, 9973),
                    b=rng.randrange(11, 9973),
                    m=rng.choice((7, 11, 13, 101, 997, 10007)),
                    d=rng.choice((2, 3, 5, 8, 13)),
                    p=rng.choice((2, 3, 4)),
                    t=rng.randrange(1, 500),
                    key=f"entry_{rng.randrange(10**6):06d}",
                )
            )
    body.append("    return registry")
    body.append("")
    return f"{name} = {len(body)}\n" + "\n".join(body)


#: Words that encode to exactly one token each under `o200k_base` when preceded by a
#: space. Discovered at import by probing a deterministic candidate list, so the pool is
#: reproducible and never depends on a random draw. A comment built from these lands on
#: an exact token count, which is what lets the corpus hit a target precisely instead of
#: approximately.
def _single_token_words(encoding: tiktoken.Encoding, limit: int = 64) -> tuple[str, ...]:
    words: list[str] = []
    seen: set[str] = set()
    for length in range(3, 10):
        for first in "abcdefghijklmnopqrstuvwxyz":
            for second in "aeiou":
                for third in "bcdfgklmnpqrstvwxyz":
                    word = first + second + third + "a" * (length - 3)
                    if word in seen:
                        continue
                    seen.add(word)
                    if len(encoding.encode(" " + word)) == 1:
                        words.append(word)
                        if len(words) >= limit:
                            return tuple(words)
    if not words:  # pragma: no cover - defensive
        raise RuntimeError("no single-token words found for the configured tokenizer")
    return tuple(words)


def _pad_comment(
    tokens_needed: int,
    encoding: tiktoken.Encoding,
    words: tuple[str, ...],
) -> str:
    """A comment encoding to exactly ``tokens_needed`` tokens.

    Each padded word is a single token, so length grows one token at a time and a short
    measured search lands exactly. The sizes differ by a fixed prefix cost depending on
    whether the line is terminated, so the search starts from both forms rather than
    assuming one. Padded with a comment rather than more code so the final tokens of the
    haystack carry no behaviour for the model to reason about.
    """
    if tokens_needed <= 0:
        return ""
    # `count_tokens("# a\n") == 3`: `#`, ` a`, `\n` each cost one token, so an unterminated
    # comment costs 1 + words and a terminated one costs 2 + words. Both are verified
    # rather than assumed, because that split is a property of the tokenizer, not a
    # guarantee, and a wrong assumption here silently mislabels every context size.
    for newline, overhead in (("\n", 2), ("", 1)):
        count = tokens_needed - overhead
        if count < 0:
            continue
        body = " ".join(words[i % len(words)] for i in range(count))
        candidate = f"# {body}{newline}" if body else f"#{newline}"
        if count_tokens(candidate, encoding) == tokens_needed:
            return candidate
    raise RuntimeError(
        f"no single-token comment encodes to exactly {tokens_needed} tokens"
    )


def build_filler(
    target_tokens: int,
    encoding: tiktoken.Encoding,
    *,
    seed: int = 42,
    module_tokens: int = 900,
) -> FillerCorpus:
    """Build filler code encoding to exactly ``target_tokens``.

    Bisects on statements-per-module rather than token count because the relationship is
    monotonic and cheap to evaluate; the exact landing is then closed with a comment.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    # Per-module PRNGs are derived from this seed inside `assemble`, so two corpora built
    # with the same seed are identical and two built with different seeds are not.
    lines_per_module = max(1, module_tokens // 12)

    def assemble(modules: int, lines: int) -> tuple[str, list[str]]:
        parts: list[str] = []
        names: list[str] = []
        for index in range(modules):
            source = _module_text(index, lines, random.Random(seed + index))
            parts.append(source)
            names.append(f"module_{index:04d}")
        return "\n".join(parts), names

    # Grow modules until the corpus is within one module of the target, then trim lines.
    modules = 1
    while True:
        text, names = assemble(modules, lines_per_module)
        size = count_tokens(text, encoding)
        if size >= target_tokens or modules > 4000:
            break
        modules += 1

    low, high = 1, lines_per_module
    while low < high:
        middle = (low + high + 1) // 2
        text, names = assemble(modules, middle)
        if count_tokens(text, encoding) <= target_tokens:
            low = middle
        else:
            high = middle - 1

    text, names = assemble(modules, low)
    size = count_tokens(text, encoding)
    # Close the gap with a comment, which cannot change any module's behaviour.
    remaining = target_tokens - size
    if remaining:
        text += _pad_comment(remaining, encoding, _single_token_words(encoding))
        size = count_tokens(text, encoding)

    if size != target_tokens:  # pragma: no cover - defensive
        raise RuntimeError(
            f"filler landed at {size} tokens, target {target_tokens}; "
            "the pad comment could not close the gap"
        )
    compile(text, "<filler>", "exec")
    return FillerCorpus(text=text, tokens=size, modules=names)


__all__ = ["FillerCorpus", "build_filler", "count_tokens", "encoding_for"]