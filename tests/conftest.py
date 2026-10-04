"""Shared pytest wiring: parametrise `cell` over the whole matrix.

The matrix itself lives in `matrix.py`; only the hook belongs here, since pytest discovers
this file and nothing else.
"""

from __future__ import annotations

import pytest
from matrix import MatrixCell, matrix


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise any test asking for `cell` over every (model, harness) pair.

    A test that takes `cell` runs once per cell, so it is asserted against each local model
    in the matrix through each harness, from that harness's own shipped config.
    """
    if "cell" in metafunc.fixturenames:
        cells: list[MatrixCell] = matrix()
        metafunc.parametrize("cell", cells, ids=[cell.id for cell in cells])
