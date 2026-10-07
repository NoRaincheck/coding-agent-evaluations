"""Needle-in-a-haystack, scored by parsing rather than by string matching.

A retrieval probe asks the model to recover one specific definition from a large body of
plausible code. The score is deliberately unforgiving:

* **Literal needles** (a constant, a specific string) are matched exactly, after
  normalising whitespace and quote style. A near miss is a miss, because "the model
  recalled the shape of the answer" is precisely the failure mode a long context is
  supposed to expose.
* **Structural needles** (a function signature) are matched by parsing the model's reply
  into an AST and comparing the recovered signature to the one planted. Asking a model to
  reproduce a signature invites paraphrase; parsing it back removes the question of
  whether "def calculate_tax(subtotal, rate=0.2)" counts, and forces the comparison to be
  about the signature itself.

Depth is recorded for every probe. A model that only retrieves from the start of a
context has a failure mode that a single pass rate hides completely, and the per-depth
breakdown is what makes that visible.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass, replace
from typing import Any

_LITERAL_KINDS = {
    "secret_constant": "SECRET_API_KEY",
    "magic_string": "RELEASE_CHANNEL",
    "function_contract": "DOCS_URL",
}


@dataclass(frozen=True)
class Needle:
    """One planted definition and the question that recovers it."""

    kind: str
    #: The module the needle is planted in, e.g. ``module_0037.py``.
    module: str
    #: 0.0 = immediately after the instruction, 1.0 = at the very end.
    depth: float
    name: str
    answer: str
    question: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "module": self.module,
            "depth": self.depth,
            "name": self.name,
            "answer": self.answer,
            "question": self.question,
        }


def make_needle(
    rng: random.Random,
    *,
    module: str,
    depth: float,
    encoding_tokens: int = 0,
) -> Needle:
    """Plant one needle. The answer is random so it cannot be guessed or memorized."""
    kind = rng.choice(sorted(_LITERAL_KINDS))
    name = _LITERAL_KINDS[kind]
    if kind == "secret_constant":
        value = f"sk_test_{rng.randrange(10**11):011d}"
        answer = value
        question = (
            f"Read the provided Python modules. In `{module}`, a module-level constant "
            f"named {name} is assigned a string value. Reply with that value only, "
            "copied exactly, and nothing else."
        )
    elif kind == "magic_string":
        value = f"release-{rng.choice(('canary', 'nightly', 'preview'))}-{rng.randrange(1000, 9999)}"
        answer = value
        question = (
            f"Read the provided Python modules. In `{module}` there is a constant named "
            f"{name} holding a string. Reply with that string exactly, and nothing else."
        )
    else:
        value = f"https://docs.example.invalid/{rng.randrange(10000, 99999)}/reference"
        answer = value
        question = (
            f"Read the provided Python modules. In `{module}` a constant named {name} "
            "holds a URL. Reply with the URL exactly, and nothing else."
        )
    return Needle(
        kind=kind,
        module=module,
        depth=depth,
        name=name,
        answer=answer,
        question=question,
    )


def signature_needle(rng: random.Random, *, module: str, depth: float) -> Needle:
    """A needle scored by AST comparison rather than string equality.

    Asking for a reproduced signature invites small reformulations, so the planted
    definition and the model's reply are both parsed and their signatures compared
    structurally. `def calculate_tax(subtotal: float, rate: float = 0.2) -> float:` and
    a `def calculate_tax(subtotal, rate=0.2):` reply agree on names, order and arity,
    which is what "did it find the definition" actually means.
    """
    name = f"calculate_tax_{rng.randrange(1000, 9999)}"
    args = [f"subtotal_{rng.randrange(100, 999)}: float"]
    rate = round(rng.uniform(0.02, 0.35), 3)
    args.append(f"rate: float = {rate}")
    return_ann = rng.choice(("float", "Decimal"))
    planted = f"def {name}({', '.join(args)}) -> {return_ann}:"
    return Needle(
        kind="signature",
        module=module,
        depth=depth,
        name=name,
        answer=planted,
        question=(
            f"Read the provided Python modules. In `{module}` there is a function whose "
            "definition you must reproduce. Reply with the `def` line of that function "
            "exactly as written, and nothing else."
        ),
    )


@dataclass(frozen=True)
class Placement:
    """Where a needle actually landed.

    The module it went into is reported rather than assumed: depth is resolved against the
    corpus's real module boundaries, and a report that says "depth 0.5" without saying
    which file it landed in cannot be reproduced or audited.
    """

    text: str
    module: str
    module_index: int
    module_count: int

    @property
    def actual_depth(self) -> float:
        if self.module_count <= 1:
            return 0.0
        return self.module_index / (self.module_count - 1)


def plant(corpus: str, needle: Needle) -> Placement:
    """Insert the needle into the filler at the requested depth.

    Depth is measured over the corpus's modules, not its characters, so 0.5 lands between
    two modules rather than possibly splitting one -- which would put the answer in a
    syntactically broken file.
    """
    boundaries = [
        (match.start(), match.group(0).split(" ")[0])
        for match in re.finditer(r"(?m)^module_\d{4} = ", corpus)
    ]
    if not boundaries:
        return Placement(
            text=f"{corpus}\n{needle_block(needle)}\n",
            module=needle.module,
            module_index=0,
            module_count=1,
        )
    index = min(len(boundaries) - 1, max(0, round(needle.depth * (len(boundaries) - 1))))
    position, module = boundaries[index]
    placed = replace(needle, module=f"{module}.py")
    return Placement(
        text=f"{corpus[:position]}{needle_block(placed)}\n{corpus[position:]}",
        module=placed.module,
        module_index=index,
        module_count=len(boundaries),
    )


def needle_block(needle: Needle) -> str:
    """The code that plants the needle."""
    if needle.kind == "signature":
        signature = needle.answer
        return (
            f"# {needle.module}\n"
            f"{signature}\n"
            "    return round(subtotal * (1.0 + rate), 2)\n"
        )
    return f"# {needle.module}\n{needle.name} = {needle.answer!r}\n"


def _normalise(text: str) -> str:
    """Strip the presentation differences that are not retrieval differences."""
    cleaned = text.strip()
    # A fenced block, however it is labelled.
    fenced = re.search(r"```(?:\w+)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    cleaned = cleaned.strip().strip("`").strip()
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r"^[\"'`]|[\"'`]$", "", cleaned)
    # A leading lead-in is packaging, not retrieval: "the value is X", "SECRET_API_KEY: X".
    cleaned = re.sub(
        r"^(?:the\s+\S+\s+is|value|answer|it\s+is)\b[\s:=-]*", "", cleaned, flags=re.IGNORECASE
    ).strip()
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _contains_value(text: str, value: str) -> bool:
    """Whether ``value`` appears in ``text`` as a whole token sequence.

    Exactness belongs on the value, not on the sentence around it. A reply of
    ``The value is `sk_test_1`.`` has retrieved the needle; a reply of ``sk_test_1sk_test_1``
    has not, because the boundaries do not line up. Comparing the whole normalised reply
    alone would score the first as a miss and would be measuring the model's packaging.
    """
    pattern = re.escape(value)
    # Boundaries that are not part of a URL or an identifier: whitespace or punctuation.
    return re.search(rf"(?<![\w./-]){pattern}(?![\w/-])", text) is not None


def _signature_of(source: str) -> dict[str, Any] | None:
    """Parse a reply and extract a comparable function signature.

    Models wrap the answer in a fence, prefix it with "Here is", or emit the whole
    function. Only the `def` line carries retrieval information, so the reply is parsed
    leniently and reduced to the signature itself.
    """
    candidates = re.findall(r"def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:\n]+)?:", source)
    if not candidates:
        return None
    # The `def` line parses on its own as a module-level statement, so it is parsed that
    # way. Wrapping it inside another function instead would make the wrapper the
    # enclosing `FunctionDef` and every signature would compare equal to every other.
    try:
        tree = ast.parse(candidates[0] + "\n    pass\n")
    except SyntaxError:
        return None
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
    )
    if function is None:
        return None
    arguments = function.args
    return {
        "name": function.name,
        # Names and order, which is what identifies the definition. Annotations are
        # dropped: a model that recovers `float` where the source said `Decimal` has
        # still found the function.
        "positional": [argument.arg for argument in arguments.args],
        "defaults": len(arguments.defaults),
        "kwonly": [argument.arg for argument in arguments.kwonlyargs],
    }


def _planted_signature(needle: Needle) -> dict[str, Any] | None:
    return _signature_of(needle.answer)


def score(needle: Needle, reply: str) -> bool:
    """Whether the reply recovered the needle.

    `unparsable` is scored as a failure on purpose. A reply that cannot be parsed did not
    demonstrate retrieval, and counting it as a near-miss would make the pass rate a
    measure of the grader's tolerance rather than of the model.
    """
    if needle.kind == "signature":
        found = _signature_of(reply)
        if found is None:
            return False
        return found == _planted_signature(needle)
    return _normalise(reply) == _normalise(needle.answer) or _contains_value(
        _normalise(reply), _normalise(needle.answer)
    )


def extract_answer(reply: str) -> str:
    """The model's reply, reduced to the part that was scored.

    Kept in the report so a failure can be read: "wrong value" and "answered something
    else entirely" are different model behaviours and the raw text distinguishes them.
    """
    return reply.strip()[:2000]


__all__ = [
    "Needle",
    "Placement",
    "extract_answer",
    "make_needle",
    "needle_block",
    "plant",
    "score",
    "signature_needle",
]