"""Topology grouping tests."""

from __future__ import annotations

import copy

import numpy as np

from simudep.genome.builtin import tetrapod
from simudep.genome.canonical import topology_hash
from simudep.genome.mutation import MutationRates, mutate
from simudep.genome.random_init import random_genome
from simudep.genome.types import Genome, Joint, Segment
from simudep.sim.grouping import group_by_topology


def _add_one_leaf_to(genome: Genome) -> None:
    """In-place: append a fresh leaf segment to the first child of the root."""

    leaf = Segment(
        size=(0.04, 0.04, 0.04),
        joint=Joint(
            axis=(0.0, 1.0, 0.0),
            anchor_parent=(0.0, 0.0, 0.05),
            range=(-1.0, 1.0),
            kp=10.0,
            kd=0.5,
        ),
    )
    genome.root.children[0].children.append(leaf)


def test_single_topology_yields_one_group() -> None:
    pop = [tetrapod() for _ in range(5)]
    groups = group_by_topology(pop)
    assert len(groups) == 1
    g = groups[0]
    assert g.indices == (0, 1, 2, 3, 4)
    assert g.size == 5
    assert g.topo_hash == topology_hash(pop[0])


def test_two_topologies_preserve_indices_and_order() -> None:
    g_a = tetrapod()
    g_b = copy.deepcopy(tetrapod())
    _add_one_leaf_to(g_b)

    pop = [g_a, g_b, copy.deepcopy(g_a), copy.deepcopy(g_b), copy.deepcopy(g_a)]
    groups = group_by_topology(pop)
    assert len(groups) == 2

    # Groups appear in first-seen order: tetrapod first, then the longer one.
    assert groups[0].topo_hash == topology_hash(g_a)
    assert groups[0].indices == (0, 2, 4)
    assert groups[1].topo_hash == topology_hash(g_b)
    assert groups[1].indices == (1, 3)


def test_groups_partition_input_exactly() -> None:
    """Every input index must appear in exactly one group."""

    rng = np.random.default_rng(0)
    pop = [random_genome(rng) for _ in range(32)]
    groups = group_by_topology(pop)

    seen: set[int] = set()
    for grp in groups:
        for idx in grp.indices:
            assert idx not in seen, "index appears in two groups"
            seen.add(idx)
    assert seen == set(range(len(pop)))


def test_continuous_mutations_keep_individuals_in_same_group() -> None:
    """A continuous-only mutation chain must keep every offspring grouped with the parent."""

    rng = np.random.default_rng(7)
    base = tetrapod()
    rates = MutationRates(p_add_leaf=0.0, p_remove_leaf=0.0)
    pop = [base] + [mutate(base, rng, rates) for _ in range(15)]
    groups = group_by_topology(pop)
    assert len(groups) == 1
    assert groups[0].size == 16
