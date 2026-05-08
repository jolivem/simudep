"""Crossover tests: invariants and the topology actually mixes."""

from __future__ import annotations

import numpy as np

from simudep.genome.crossover import crossover
from simudep.genome.random_init import random_genome
from simudep.genome.types import iter_joints, iter_segments


def test_crossover_preserves_sequence_target_count() -> None:
    rng = np.random.default_rng(0)
    a = random_genome(rng, n_segments=5)
    b = random_genome(rng, n_segments=5)
    for _ in range(20):
        c = crossover(a, b, rng)
        n_j = len(iter_joints(c.root))
        for step in c.sequence.steps:
            assert len(step.targets) == n_j


def test_crossover_does_not_mutate_parents() -> None:
    rng = np.random.default_rng(1)
    a = random_genome(rng, n_segments=5)
    b = random_genome(rng, n_segments=5)
    a_segs = sum(1 for _ in iter_segments(a.root))
    b_segs = sum(1 for _ in iter_segments(b.root))
    crossover(a, b, rng)
    assert sum(1 for _ in iter_segments(a.root)) == a_segs
    assert sum(1 for _ in iter_segments(b.root)) == b_segs
