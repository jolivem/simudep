"""Random genome generation.

A random creature is built by:
  1. Sampling a root box.
  2. Repeatedly attaching a new "limb-like" box to one of the existing
     segments, on the center of one of its 6 faces. The new segment's
     long axis aligns with the attachment direction; the joint axis is
     perpendicular to it (so the limb can swing around).
  3. Sampling a fixed-length cyclic sequence of joint targets.

The initial root height is set so the lowest geom point sits just above the
ground (z = epsilon) at rest.
"""

from __future__ import annotations

import math

import numpy as np

from simudep.genome.types import (
    Genome,
    Joint,
    Segment,
    Sequence,
    SequenceStep,
    iter_joints,
)

# Sensible bounds. Kept conservative so random creatures don't blow up MJX.
_DENSITY = 800.0  # kg/m^3
MAX_SEGMENTS = 12  # global cap on tree size (also enforced in mutation/crossover)
_ROOT_HALF_RANGE = (0.05, 0.14)
_LIMB_LENGTH_RANGE = (0.07, 0.18)
_LIMB_RADIUS_RANGE = (0.020, 0.045)
_KP_RANGE = (12.0, 35.0)
_KD_RANGE = (0.5, 1.6)
_JOINT_LIMIT = math.radians(80.0)
_TARGET_RANGE = math.radians(70.0)
_N_SEQUENCE_STEPS = 8
_STEP_DURATION_RANGE = (0.18, 0.40)
_GROUND_CLEARANCE = 0.005


def random_genome(
    rng: np.random.Generator | None = None,
    *,
    n_segments: int | None = None,
    name: str | None = None,
) -> Genome:
    """Generate a random tree-shaped genome with a cyclic sequence."""

    rng = _ensure_rng(rng)

    if n_segments is None:
        n_segments = int(rng.integers(3, 9))  # 3..8 segments total
    n_segments = max(2, min(MAX_SEGMENTS, n_segments))

    root = _random_root(rng)
    segments: list[Segment] = [root]

    for _ in range(n_segments - 1):
        parent = segments[int(rng.integers(0, len(segments)))]
        child = _random_limb(parent, rng)
        parent.children.append(child)
        segments.append(child)

    n_joints = len(iter_joints(root))
    sequence = _random_sequence(n_joints, rng)
    initial_root_pos = _initial_root_pos(root)

    if name is None:
        name = f"rnd_{int(rng.integers(0, 1 << 24)):06x}"

    return Genome(
        root=root,
        sequence=sequence,
        initial_root_pos=initial_root_pos,
        id=name,
    )


# -- internals ----------------------------------------------------------


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _random_color(rng: np.random.Generator) -> tuple[float, float, float, float]:
    rgb = rng.uniform(0.30, 0.85, size=3)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _box_mass(half_extents: tuple[float, float, float]) -> float:
    volume = 8.0 * half_extents[0] * half_extents[1] * half_extents[2]
    return float(_DENSITY * volume)


def _random_root(rng: np.random.Generator) -> Segment:
    half = tuple(float(rng.uniform(*_ROOT_HALF_RANGE)) for _ in range(3))
    return Segment(
        size=half,  # type: ignore[arg-type]
        geom_pos=(0.0, 0.0, 0.0),
        mass=_box_mass(half),  # type: ignore[arg-type]
        color=_random_color(rng),
    )


def _random_limb(parent: Segment, rng: np.random.Generator) -> Segment:
    """Attach a limb-shaped box to the center of a random face of `parent`."""

    face_axis = int(rng.integers(0, 3))
    face_sign = 1.0 if rng.random() < 0.5 else -1.0

    # Joint anchor on the parent: center of a face, in the parent's local frame.
    anchor = [0.0, 0.0, 0.0]
    anchor[face_axis] = parent.geom_pos[face_axis] + face_sign * parent.size[face_axis]

    # Limb shape: long along `face_axis`, narrow on the other two.
    length = float(rng.uniform(*_LIMB_LENGTH_RANGE))
    radius = float(rng.uniform(*_LIMB_RADIUS_RANGE))
    half_extents = [radius, radius, radius]
    half_extents[face_axis] = length
    size = tuple(half_extents)

    # Geom centered along the limb so it extends outward from the joint.
    geom_pos = [0.0, 0.0, 0.0]
    geom_pos[face_axis] = face_sign * length

    # Joint axis: one of the two axes perpendicular to the attachment direction.
    perp_axes = [a for a in (0, 1, 2) if a != face_axis]
    joint_axis_idx = int(rng.choice(perp_axes))
    axis = [0.0, 0.0, 0.0]
    axis[joint_axis_idx] = 1.0

    joint = Joint(
        axis=tuple(axis),  # type: ignore[arg-type]
        anchor_parent=tuple(anchor),  # type: ignore[arg-type]
        range=(-_JOINT_LIMIT, _JOINT_LIMIT),
        kp=float(rng.uniform(*_KP_RANGE)),
        kd=float(rng.uniform(*_KD_RANGE)),
    )

    return Segment(
        size=size,  # type: ignore[arg-type]
        geom_pos=tuple(geom_pos),  # type: ignore[arg-type]
        mass=_box_mass(size),  # type: ignore[arg-type]
        color=_random_color(rng),
        joint=joint,
    )


def _random_sequence(n_joints: int, rng: np.random.Generator) -> Sequence:
    steps = []
    for _ in range(_N_SEQUENCE_STEPS):
        duration = float(rng.uniform(*_STEP_DURATION_RANGE))
        targets = tuple(float(rng.uniform(-_TARGET_RANGE, _TARGET_RANGE)) for _ in range(n_joints))
        steps.append(SequenceStep(duration=duration, targets=targets))
    cycle = sum(s.duration for s in steps)
    return Sequence(cycle_duration=cycle, steps=tuple(steps))


def _initial_root_pos(root: Segment) -> tuple[float, float, float]:
    """Place the root so that the lowest geom corner sits at z = clearance.

    All bodies are aligned with the root's frame at rest (no rotations in
    the genome model), so each segment's frame z = chained sum of its
    ancestors' joint anchor z components.
    """

    lowest = math.inf
    for seg, frame_z in _iter_segment_frame_z(root):
        bottom = frame_z + seg.geom_pos[2] - seg.size[2]
        if bottom < lowest:
            lowest = bottom
    if not math.isfinite(lowest):
        lowest = 0.0
    return (0.0, 0.0, _GROUND_CLEARANCE - lowest)


def _iter_segment_frame_z(root: Segment):
    """Yield (segment, frame_z_in_root_frame) for every segment in the tree."""

    stack = [(root, 0.0)]
    while stack:
        seg, z = stack.pop()
        yield seg, z
        for child in seg.children:
            assert child.joint is not None
            stack.append((child, z + child.joint.anchor_parent[2]))


__all__ = [
    "random_genome",
]
