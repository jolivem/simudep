"""Mutation tests: invariants, identity, and that topology mutations re-align sequences."""

from __future__ import annotations

import numpy as np

from simudep.genome.mutation import MutationRates, mutate
from simudep.genome.random_init import random_genome
from simudep.genome.types import iter_joints, iter_segments


def _n_segments(g) -> int:
    return sum(1 for _ in iter_segments(g.root))


def _n_joints(g) -> int:
    return len(iter_joints(g.root))


def test_mutation_preserves_sequence_target_count() -> None:
    rng = np.random.default_rng(0)
    g0 = random_genome(rng, n_segments=5)
    for _ in range(50):
        g = mutate(g0, rng)
        n_j = _n_joints(g)
        for step in g.sequence.steps:
            assert len(step.targets) == n_j


def test_mutation_returns_a_distinct_object() -> None:
    rng = np.random.default_rng(0)
    g0 = random_genome(rng, n_segments=4)
    g1 = mutate(g0, rng)
    assert g1 is not g0
    assert g1.root is not g0.root
    # Modifying g1 must not affect g0.
    g1.root.children.clear()
    assert len(g0.root.children) > 0


def test_topological_only_mutations_change_segment_count_eventually() -> None:
    """With enough mutations, add/remove leaf must change the structural size."""
    rng = np.random.default_rng(7)
    g = random_genome(rng, n_segments=4)
    rates = MutationRates(
        p_size=0.0, p_kp_kd=0.0, p_targets=0.0, p_durations=0.0,
        p_add_leaf=1.0, p_remove_leaf=0.0,
    )
    sizes = {_n_segments(g)}
    for _ in range(20):
        g = mutate(g, rng, rates)
        sizes.add(_n_segments(g))
    assert len(sizes) > 1, "add_leaf with p=1 should grow the tree at least once"
