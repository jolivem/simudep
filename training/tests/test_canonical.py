"""Topology hash and signature tests."""

from __future__ import annotations

import copy

import numpy as np

from simudep.genome.builtin import tetrapod
from simudep.genome.canonical import topology_hash, topology_signature
from simudep.genome.mutation import MutationRates, mutate
from simudep.genome.random_init import random_genome
from simudep.genome.types import Genome, Joint, Segment, Sequence, SequenceStep


def _trivial_genome(root: Segment) -> Genome:
    seq = Sequence(cycle_duration=1.0, steps=(SequenceStep(duration=1.0, targets=()),))
    return Genome(root=root, sequence=seq)


def _leaf_segment() -> Segment:
    return Segment(size=(0.05, 0.05, 0.05))


def _trivial_joint() -> Joint:
    return Joint(
        axis=(1.0, 0.0, 0.0),
        anchor_parent=(0.0, 0.0, 0.0),
        range=(-1.0, 1.0),
        kp=10.0,
        kd=0.5,
    )


def test_lone_root_signature() -> None:
    g = _trivial_genome(_leaf_segment())
    assert topology_signature(g) == "()"


def test_one_child_signature() -> None:
    root = _leaf_segment()
    child = _leaf_segment()
    child.joint = _trivial_joint()
    root.children.append(child)
    g = _trivial_genome(root)
    assert topology_signature(g) == "(())"


def test_tetrapod_signature_is_root_with_four_leaves() -> None:
    g = tetrapod()
    # root + 4 leaf legs → "(" + "()()()()" + ")"
    assert topology_signature(g) == "(()()()())"


def test_topology_hash_is_stable_across_calls() -> None:
    g = tetrapod()
    assert topology_hash(g) == topology_hash(tetrapod())
    assert len(topology_hash(g)) == 16


def test_continuous_mutations_preserve_hash() -> None:
    """Size/mass/kp/kd/sequence mutations must NOT change the topology hash."""

    rng = np.random.default_rng(0)
    base = tetrapod()
    rates = MutationRates(p_add_leaf=0.0, p_remove_leaf=0.0)
    for _ in range(20):
        m = mutate(base, rng, rates)
        assert topology_hash(m) == topology_hash(base)


def test_structural_mutations_do_change_hash() -> None:
    """Adding a leaf must change the hash for at least some draws."""

    rng = np.random.default_rng(0)
    base = tetrapod()
    rates = MutationRates(
        p_size=0.0,
        p_kp_kd=0.0,
        p_targets=0.0,
        p_durations=0.0,
        p_add_leaf=1.0,
        p_remove_leaf=0.0,
    )
    base_h = topology_hash(base)
    new_hashes = {topology_hash(mutate(base, rng, rates)) for _ in range(10)}
    # All draws add at least one leaf, so all hashes must differ from base.
    assert base_h not in new_hashes


def test_random_population_has_diverse_hashes() -> None:
    """Sanity check: random_genome yields several distinct topologies in 32 draws."""

    rng = np.random.default_rng(42)
    hashes = {topology_hash(random_genome(rng)) for _ in range(32)}
    # Highly unlikely to collapse to a single bucket — but at minimum >1.
    assert len(hashes) > 1


def test_deepcopy_preserves_hash() -> None:
    g = tetrapod()
    assert topology_hash(g) == topology_hash(copy.deepcopy(g))
