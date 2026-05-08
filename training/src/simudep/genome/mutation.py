"""Genome mutations.

Two families:

  * **Continuous** — perturb existing scalars (segment sizes, joint gains,
    sequence target angles and durations). Topology unchanged, so the DFS
    ordering and sequence target arity are preserved.

  * **Topological** — add a new limb on a random segment, or remove an
    existing leaf segment. After such an edit we re-derive the DFS joint
    ordering and patch every sequence step to insert/drop the corresponding
    target value. Joint object identity (`is`) is used to locate the
    inserted/removed joint, since we don't deep-copy joints inside a
    mutation step (only the caller does, once, before mutating).

`mutate(genome, rng)` deep-copies the input then applies a small random
batch of mutations and returns the new genome.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

# Internal helpers shared with the random initializer (limb sampling and
# initial-position recomputation). Treated as package-internal API.
from simudep.genome.random_init import MAX_SEGMENTS, _initial_root_pos, _random_limb
from simudep.genome.types import (
    Genome,
    Joint,
    Segment,
    Sequence,
    SequenceStep,
    iter_joints,
    iter_segments,
)

_TARGET_LIMIT = math.radians(80.0)


@dataclass(frozen=True)
class MutationRates:
    """Probabilities applied independently per mutation."""

    p_size: float = 0.6
    p_kp_kd: float = 0.6
    p_targets: float = 0.8
    p_durations: float = 0.4
    p_add_leaf: float = 0.15
    p_remove_leaf: float = 0.10

    sigma_size_log: float = 0.10
    sigma_kp_log: float = 0.20
    sigma_kd_log: float = 0.20
    sigma_target: float = math.radians(15.0)
    sigma_duration_log: float = 0.15


def mutate(
    genome: Genome,
    rng: np.random.Generator,
    rates: MutationRates | None = None,
) -> Genome:
    """Return a mutated deep copy of `genome`."""

    rates = rates or MutationRates()
    g = copy.deepcopy(genome)

    if rng.random() < rates.p_size:
        _perturb_segment_sizes(g, rng, rates)
    if rng.random() < rates.p_kp_kd:
        _perturb_joint_gains(g, rng, rates)
    if rng.random() < rates.p_targets:
        _perturb_sequence_targets(g, rng, rates)
    if rng.random() < rates.p_durations:
        _perturb_sequence_durations(g, rng, rates)

    if rng.random() < rates.p_add_leaf:
        _add_leaf(g, rng)
    if rng.random() < rates.p_remove_leaf:
        _remove_leaf(g, rng)

    g.initial_root_pos = _initial_root_pos(g.root)
    return g


# -- continuous mutations -----------------------------------------------


def _perturb_segment_sizes(g: Genome, rng: np.random.Generator, r: MutationRates) -> None:
    for seg in iter_segments(g.root):
        new_size = tuple(
            float(np.clip(s * math.exp(rng.normal(0.0, r.sigma_size_log)), 0.012, 0.30))
            for s in seg.size
        )
        seg.size = new_size  # type: ignore[assignment]
        # Keep mass roughly proportional to volume (constant density).
        volume = 8.0 * new_size[0] * new_size[1] * new_size[2]
        seg.mass = max(0.02, 800.0 * volume)


def _perturb_joint_gains(g: Genome, rng: np.random.Generator, r: MutationRates) -> None:
    """Joints are frozen dataclasses, so we replace them in-place on the parent."""

    def _walk(seg: Segment) -> None:
        for child in seg.children:
            assert child.joint is not None
            old = child.joint
            kp = float(np.clip(old.kp * math.exp(rng.normal(0.0, r.sigma_kp_log)), 5.0, 80.0))
            kd = float(np.clip(old.kd * math.exp(rng.normal(0.0, r.sigma_kd_log)), 0.1, 4.0))
            child.joint = Joint(
                axis=old.axis,
                anchor_parent=old.anchor_parent,
                range=old.range,
                kp=kp,
                kd=kd,
            )
            _walk(child)

    _walk(g.root)


def _perturb_sequence_targets(g: Genome, rng: np.random.Generator, r: MutationRates) -> None:
    new_steps = []
    for step in g.sequence.steps:
        targets = tuple(
            float(np.clip(t + rng.normal(0.0, r.sigma_target), -_TARGET_LIMIT, _TARGET_LIMIT))
            for t in step.targets
        )
        new_steps.append(SequenceStep(duration=step.duration, targets=targets))
    g.sequence = Sequence(
        cycle_duration=sum(s.duration for s in new_steps),
        steps=tuple(new_steps),
    )


def _perturb_sequence_durations(g: Genome, rng: np.random.Generator, r: MutationRates) -> None:
    new_steps = []
    for step in g.sequence.steps:
        scale = math.exp(rng.normal(0.0, r.sigma_duration_log))
        d = float(np.clip(step.duration * scale, 0.08, 0.8))
        new_steps.append(SequenceStep(duration=d, targets=step.targets))
    g.sequence = Sequence(
        cycle_duration=sum(s.duration for s in new_steps),
        steps=tuple(new_steps),
    )


# -- topological mutations ----------------------------------------------


def _add_leaf(g: Genome, rng: np.random.Generator) -> None:
    segments = list(iter_segments(g.root))
    if len(segments) >= MAX_SEGMENTS:
        return  # cap to keep simulations tractable
    parent = segments[int(rng.integers(0, len(segments)))]
    new_seg = _random_limb(parent, rng)
    parent.children.append(new_seg)

    # Locate the new joint in the new DFS order via object identity.
    joints_after = iter_joints(g.root)
    assert new_seg.joint is not None
    new_pos = _index_of_identity(joints_after, new_seg.joint)

    new_target = float(rng.uniform(-_TARGET_LIMIT, _TARGET_LIMIT))
    new_steps = []
    for step in g.sequence.steps:
        targets = list(step.targets)
        targets.insert(new_pos, new_target + float(rng.normal(0.0, math.radians(10.0))))
        new_steps.append(SequenceStep(duration=step.duration, targets=tuple(targets)))
    g.sequence = Sequence(
        cycle_duration=sum(s.duration for s in new_steps),
        steps=tuple(new_steps),
    )


def _remove_leaf(g: Genome, rng: np.random.Generator) -> None:
    leaves = [
        s for s in iter_segments(g.root) if not s.children and s.joint is not None
    ]  # excludes the root since root has joint=None
    if not leaves:
        return
    if sum(1 for _ in iter_segments(g.root)) <= 2:
        return  # keep at least 1 limb so there's something to control

    leaf = leaves[int(rng.integers(0, len(leaves)))]
    parent = _find_parent(g.root, leaf)
    assert parent is not None
    assert leaf.joint is not None

    joints_before = iter_joints(g.root)
    old_pos = _index_of_identity(joints_before, leaf.joint)

    parent.children.remove(leaf)

    new_steps = []
    for step in g.sequence.steps:
        targets = list(step.targets)
        del targets[old_pos]
        new_steps.append(SequenceStep(duration=step.duration, targets=tuple(targets)))
    g.sequence = Sequence(
        cycle_duration=sum(s.duration for s in new_steps),
        steps=tuple(new_steps),
    )


# -- helpers ------------------------------------------------------------


def _find_parent(root: Segment, target: Segment) -> Segment | None:
    if target is root:
        return None
    stack = [root]
    while stack:
        s = stack.pop()
        for c in s.children:
            if c is target:
                return s
            stack.append(c)
    return None


def _index_of_identity(items: list[Joint], target: Joint) -> int:
    for i, j in enumerate(items):
        if j is target:
            return i
    raise AssertionError("joint not found by identity")


__all__ = [
    "MutationRates",
    "mutate",
]
