"""Genome ↔ JSON roundtrip tests."""

from __future__ import annotations

from simudep.genome.builtin import tetrapod
from simudep.genome.types import iter_joints, iter_segments
from simudep.io.genome_json import genome_dumps, genome_loads


def test_tetrapod_roundtrip_preserves_structure() -> None:
    g0 = tetrapod()
    g1 = genome_loads(genome_dumps(g0))

    assert g1.id == g0.id
    assert g1.version == g0.version
    assert g1.initial_root_pos == g0.initial_root_pos
    assert len(iter_segments(g1.root)) == len(iter_segments(g0.root))
    assert len(iter_joints(g1.root)) == len(iter_joints(g0.root))


def test_tetrapod_roundtrip_preserves_values() -> None:
    g0 = tetrapod()
    g1 = genome_loads(genome_dumps(g0))

    j0 = iter_joints(g0.root)
    j1 = iter_joints(g1.root)
    for a, b in zip(j0, j1, strict=True):
        assert a.axis == b.axis
        assert a.anchor_parent == b.anchor_parent
        assert a.range == b.range
        assert a.kp == b.kp
        assert a.kd == b.kd

    assert g0.sequence.cycle_duration == g1.sequence.cycle_duration
    assert len(g0.sequence.steps) == len(g1.sequence.steps)
    for s0, s1 in zip(g0.sequence.steps, g1.sequence.steps, strict=True):
        assert s0.duration == s1.duration
        assert s0.targets == s1.targets


def test_root_has_no_joint_after_roundtrip() -> None:
    g = genome_loads(genome_dumps(tetrapod()))
    assert g.root.joint is None
    for child in g.root.children:
        assert child.joint is not None
