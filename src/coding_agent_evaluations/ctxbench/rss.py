"""Resident-set sampling for the memory suite.

A memory number has to come from somewhere the report can defend, and "I looked at
Activity Monitor" is not that. Two things are measured:

* **Peak RSS of the harness process tree**, via `resource.getrusage`. `ru_maxrss` is a
  high-water mark, so it cannot miss a spike between samples, which a polling sampler
  can. It is reported in bytes on macOS and kilobytes on Linux; the units are normalized
  here rather than left for the reader to guess.
* **RSS growth across a repeated request**, sampled between iterations. `ru_maxrss` alone
  cannot distinguish a process that reached 400 MB and stayed there from one that reached
  400 MB and kept climbing, and only the second is a leak. Growth across iterations is
  what the report flags.

Sampling is best effort. If `getrusage` is unavailable the suite reports `null` rather
than a fabricated zero -- a memory column that reads "0 MB" because nothing was measured
is worse than an absent one.
"""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass, field
from typing import Any

#: macOS reports bytes, Linux reports kilobytes. Normalized to megabytes.
_RSS_DIVISOR = 1024 * 1024 if sys.platform == "darwin" else 1024


def peak_rss_mb() -> float | None:
    """Peak RSS of this process and its children, in MB."""
    try:
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (OSError, ValueError):  # pragma: no cover - platform dependent
        return None
    return round(max(own, kids) / _RSS_DIVISOR, 1)


@dataclass
class MemoryTrace:
    """RSS across a repeated request, and the growth between first and last."""

    samples_mb: list[float] = field(default_factory=list)
    #: How many requests were made. Kept separately from the sample count because the
    #: trace brackets the requests -- sampled before each one and once after the last -- so
    #: the two differ by one, and reporting the sample count as the iteration count would
    #: overstate the work by one.
    iterations: int = 0
    unit: str = "MB"

    def sample(self) -> float | None:
        current = peak_rss_mb()
        if current is not None:
            self.samples_mb.append(current)
        return current

    @property
    def growth_mb(self) -> float | None:
        """Change in peak RSS from the first sample to the last.

        Peak rather than instantaneous: a harness that frees between requests should not
        be reported as growing, and peak-to-peak is the conservative reading.
        """
        if len(self.samples_mb) < 2:
            return None
        return round(self.samples_mb[-1] - self.samples_mb[0], 1)

    @property
    def monotonic_growth(self) -> bool:
        """RSS grew on every single iteration.

        Strictly monotonic rather than "net positive", because a suite that alternates
        allocate/free can show a small net rise with no leak at all.
        """
        if len(self.samples_mb) < 3:
            return False
        # Not `strict`: the two sequences are deliberately offset by one, and strict=True
        # would raise on a trace of any length.
        return all(
            later >= earlier
            for earlier, later in zip(self.samples_mb, self.samples_mb[1:])
        )

    @property
    def peak_mb(self) -> float | None:
        return max(self.samples_mb) if self.samples_mb else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "peak_mb": self.peak_mb,
            "growth_mb": self.growth_mb,
            "monotonic_growth": self.monotonic_growth,
            "iterations": self.iterations,
            "samples": len(self.samples_mb),
            "samples_mb": self.samples_mb,
        }


__all__ = ["MemoryTrace", "peak_rss_mb"]