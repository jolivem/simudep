"""Random genome generation tests."""

from __future__ import annotations

import numpy as np

from simudep.genome.random_init import random_genome
from simudep.genome.types import iter_joints, iter_segments
from simudep.mjcf.builder import build_mjcf


def test_seeded_random_is_reproducible() -> None:
    g1 = random_genome(np.random.default_rng(42))
    g2 = random_genome(np.random.default_rng(42))
    assert len(iter_segments(g1.root)) == len(iter_segments(g2.root))
    assert len(iter_joints(g1.root)) == len(iter_joints(g2.root))
    assert g1.initial_root_pos == g2.initial_root_pos


def test_random_genome_invariants() -> None:
    rng = np.random.default_rng(0)
    for _ in range(8):
        g = random_genome(rng)
        n_joints = len(iter_joints(g.root))
        # Sequence target arity must match joint count.
        for step in g.sequence.steps:
            assert len(step.targets) == n_joints
        # The root segment is the only one without a joint.
        for seg in iter_segments(g.root):
            if seg is g.root:
                assert seg.joint is None
            else:
                assert seg.joint is not None
        # Initial root position keeps the lowest geom above ground.
        assert g.initial_root_pos[2] > 0


def test_random_genomes_build_valid_mjcf() -> None:
    """A random tree must produce an MJCF that MuJoCo accepts."""
    import mujoco

    rng = np.random.default_rng(1234)
    for _ in range(5):
        g = random_genome(rng)
        xml = build_mjcf(g)
        model = mujoco.MjModel.from_xml_string(xml)
        n = len(iter_joints(g.root))
        assert model.nu == n
        assert model.nq == 7 + n
