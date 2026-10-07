"""Context-scaling evaluation: how a model and a harness hold up as context grows.

SWE-bench answers "can this model solve the task". This answers a different question:
at 16k, 32k, 64k and 128k tokens of context, how fast does the endpoint answer, does the
model still find what was buried in the middle, does the harness silently drop what it
could not fit, and does the whole thing stay healthy under repetition.

The suites are deliberately separate so a failure names itself:

* ``prefill``    — a padded prompt with a one-word answer. Isolates context processing.
* ``generation`` — a fixed coding prompt. Measures decode throughput.
* ``needle``     — retrieval of a specific definition buried at a known depth.
* ``execution``  — real benchmark tasks graded by their own verifiers (pass@1).
* ``memory``     — the same request repeated, sampling RSS, to expose growth over time.
"""

from .config import CtxBenchConfig, CtxSuiteConfig, load_ctx_config
from .report import as_json, as_markdown
from .runner import run_ctxbench

__all__ = [
    "CtxBenchConfig",
    "CtxSuiteConfig",
    "as_json",
    "as_markdown",
    "load_ctx_config",
    "run_ctxbench",
]