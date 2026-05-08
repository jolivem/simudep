"""Rollout sanity & determinism tests (CPU)."""

from __future__ import annotations

import numpy as np

from simudep.genome.builtin import tetrapod
from simudep.sim.rollout import rollout_cpu


def test_rollout_returns_expected_shape() -> None:
    g = tetrapod()
    r = rollout_cpu(g, duration=1.0, fps=60)
    assert r.qpos.shape == (60, r.nq)
    assert r.qpos.dtype == np.float32
    assert r.fps == 60
    assert r.n_joints == 4


def test_rollout_determinism_cpu() -> None:
    g = tetrapod()
    r1 = rollout_cpu(g, duration=1.0, fps=60)
    r2 = rollout_cpu(g, duration=1.0, fps=60)
    np.testing.assert_array_equal(r1.qpos, r2.qpos)
    assert r1.fitness == r2.fitness


def test_rollout_motors_do_work() -> None:
    """Even a non-walking creature should burn energy because legs swing."""
    g = tetrapod()
    r = rollout_cpu(g, duration=1.0, fps=60)
    assert r.fitness["energy"] > 0.1
