"""Dataset loading.

Task definitions come from FrogNano's dataset sources so the pinned revisions,
image references, timeouts and instructions match the reference evaluation. This
module adds only the selection that a single-machine run needs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coding_agent_evaluations._vendor.frognano.datasets import get_dataset
from coding_agent_evaluations._vendor.frognano.datasets import (
    load_dataset as _load_dataset,
)

SUPPORTED = (
    "swebench_verified",
    "terminal_bench_2_verified",
)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    display_name: str
    source_url: str
    revision: str
    subpath: str
    agent_network_mode: str
    verifier_network_mode: str


def describe(name: str) -> DatasetInfo:
    """Metadata for a registered dataset without materializing it."""
    if name not in SUPPORTED:
        raise ValueError(
            f"dataset {name!r} is not supported by the host runtime; "
            f"available: {', '.join(SUPPORTED)}"
        )
    source, _ = get_dataset(name)
    return DatasetInfo(
        name=source.name,
        display_name=source.display_name,
        source_url=source.source_url,
        revision=source.revision,
        subpath=source.subpath,
        agent_network_mode=source.agent_network_mode,
        verifier_network_mode=source.verifier_network_mode,
    )


def load_tasks(
    name: str,
    *,
    cache_dir: Path,
    task_ids: tuple[str, ...] = (),
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load, shuffle and truncate tasks exactly like the reference runner.

    Shuffling with the same seed as FrogNano means a ``num_tasks`` smoke slice
    selects the same task subset here as it would there.
    """
    tasks = _load_dataset(
        name,
        cache_dir=cache_dir,
        image_registry=None,
        task_ids=task_ids,
        limit=None,
    )
    random.Random(seed).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


__all__ = ["SUPPORTED", "DatasetInfo", "describe", "load_tasks"]