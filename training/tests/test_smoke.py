"""Phase 0 smoke tests: imports and version metadata."""

from __future__ import annotations


def test_version() -> None:
    import simudep

    assert simudep.__version__ == "0.1.0"


def test_imports() -> None:
    import jax  # noqa: F401
    import mujoco  # noqa: F401
    import mujoco.mjx  # noqa: F401
    import numpy  # noqa: F401


def test_cli_help_runs() -> None:
    from simudep.cli.__main__ import main

    assert main([]) == 0
