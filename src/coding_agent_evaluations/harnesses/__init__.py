"""Harness adapters.

A harness is the agent loop under test. Every harness receives a
:class:`HarnessRun` (the provisioned workspace plus limits) and returns a
trajectory in the same shape, so results are comparable across harnesses.

* ``leaf``     - the reference harness from FrogNano, used unmodified.
* ``opencode`` - the opencode CLI, driven headless.
* ``pi``       - the pi CLI, driven headless.
"""

from __future__ import annotations

from .base import Harness, HarnessRun, build_harness

__all__ = ["Harness", "HarnessRun", "build_harness"]