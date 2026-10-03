"""Cross-harness evaluation of OpenAI-compatible coding models.

Runs the pinned FrogNano benchmarks (SWE-bench Verified, Terminal-Bench 2
Verified) against a local OpenAI-compatible endpoint, once per agent harness:

* ``leaf``     - the reference harness, used exactly as FrogNano ships it.
* ``opencode`` - the opencode CLI, headless.
* ``pi``       - the pi CLI, headless.

The same task, model and limits go to every harness, so the resulting
``summary.json`` files can be compared directly.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]