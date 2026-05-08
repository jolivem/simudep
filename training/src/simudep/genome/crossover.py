"""Subtree crossover.

Given two parent genomes A and B, produce an offspring that is "A with one
of its sub-trees replaced by a sub-tree taken from B". The grafted sub-tree
retains its physical shape from B but inherits A's joint at its base (so it
connects at the same anchor in the offspring as the sub-tree it is
replacing). Both parents' fixed-length sequences are sliced and spliced at
the corresponding DFS joint range.
"""

from __future__ import annotations

import copy

import numpy as np

from simudep.genome.random_init import MAX_SEGMENTS, _initial_root_pos
from simudep.genome.types import (
    Genome,
    Joint,
    Segment,
    Sequence,
    SequenceStep,
    iter_joints,
)


def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome:
    """Return a new genome by grafting a random sub-tree of B onto a copy of A."""

    child = copy.deepcopy(parent_a)

    a_candidates = _non_root_segments(child.root)
    b_candidates = _non_root_segments(parent_b.root)
    if not a_candidates or not b_candidates:
        return child  # one parent has no swappable sub-tree

    a_size_total = sum(1 for _ in _non_root_segments(child.root)) + 1  # include root
    a_target = a_candidates[int(rng.integers(0, len(a_candidates)))]
    a_replaced_size = _count_subtree_segments(a_target)
    capacity = MAX_SEGMENTS - (a_size_total - a_replaced_size)

    # Pick a B sub-tree that fits within the per-individual segment cap.
    feasible = [s for s in b_candidates if _count_subtree_segments(s) <= capacity]
    if not feasible:
        return child  # any B sub-tree would overflow the cap; bail.
    b_source = feasible[int(rng.integers(0, len(feasible)))]

    a_parent = _find_parent(child.root, a_target)
    assert a_parent is not None
    assert a_target.joint is not None
    assert b_source.joint is not None

    # Joint slice replaced in A's DFS (a_target's joint + all descendants' joints).
    full_joints_a = iter_joints(child.root)
    a_start = _identity_index(full_joints_a, a_target.joint)
    a_size = _count_subtree_joints(a_target)
    a_end = a_start + a_size

    # Joint slice taken from B (b_source's joint + descendants).
    full_joints_b = iter_joints(parent_b.root)
    b_start = _identity_index(full_joints_b, b_source.joint)
    b_size = _count_subtree_joints(b_source)
    b_end = b_start + b_size

    # Graft a deep copy of B's sub-tree under A's parent, but keep A's joint
    # so the connection geometry (axis, anchor on parent, range) is preserved.
    grafted = copy.deepcopy(b_source)
    grafted.joint = Joint(
        axis=a_target.joint.axis,
        anchor_parent=a_target.joint.anchor_parent,
        range=a_target.joint.range,
        kp=a_target.joint.kp,
        kd=a_target.joint.kd,
    )

    idx = a_parent.children.index(a_target)
    a_parent.children[idx] = grafted

    # Sequence: keep A's step durations, but in each step replace the
    # [a_start:a_end] target slice with B's [b_start:b_end] slice.
    new_steps: list[SequenceStep] = []
    for step_a, step_b in zip(parent_a.sequence.steps, parent_b.sequence.steps, strict=False):
        targets = list(step_a.targets)
        b_slice = list(step_b.targets[b_start:b_end])
        targets[a_start:a_end] = b_slice
        new_steps.append(SequenceStep(duration=step_a.duration, targets=tuple(targets)))
    child.sequence = Sequence(
        cycle_duration=sum(s.duration for s in new_steps),
        steps=tuple(new_steps),
    )

    child.initial_root_pos = _initial_root_pos(child.root)
    return child


# -- helpers ------------------------------------------------------------


def _non_root_segments(root: Segment) -> list[Segment]:
    out: list[Segment] = []

    def walk(seg: Segment) -> None:
        for c in seg.children:
            out.append(c)
            walk(c)

    walk(root)
    return out


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


def _identity_index(items: list[Joint], target: Joint) -> int:
    for i, j in enumerate(items):
        if j is target:
            return i
    raise AssertionError("joint not found by identity")


def _count_subtree_joints(seg: Segment) -> int:
    """Count joints in the sub-tree rooted at `seg` (including seg.joint)."""

    n = 1 if seg.joint is not None else 0
    for c in seg.children:
        n += _count_subtree_joints(c)
    return n


def _count_subtree_segments(seg: Segment) -> int:
    """Count segments in the sub-tree rooted at `seg` (including seg itself)."""

    n = 1
    for c in seg.children:
        n += _count_subtree_segments(c)
    return n


__all__ = ["crossover"]
