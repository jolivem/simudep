"""Genome data model.

The creature is a tree of rigid box segments connected by 1-DOF revolute
joints. The root segment is attached to the world by an implicit free joint
and therefore has no `Joint` of its own. Every other segment is attached
to its parent by a `Joint` whose anchor point lives in the parent's local
frame.

Each child segment's local frame has its origin at the joint anchor in the
parent's frame, with its axes parallel to the parent's axes. The geom
(visual + collision box) is positioned inside the segment's frame via
`geom_pos` and shaped via `size` (half-extents, MuJoCo convention).

The sequence is a cyclic schedule of joint-angle targets. The list of
target angles in each step is ordered by depth-first traversal of the
tree (skipping the root, since the root has no joint).
"""

from __future__ import annotations

from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


@dataclass(frozen=True)
class Joint:
    """A 1-DOF revolute joint connecting a child segment to its parent."""

    axis: Vec3
    """Unit vector of the rotation axis, expressed in the parent's frame."""

    anchor_parent: Vec3
    """Joint location in the parent segment's local frame."""

    range: tuple[float, float]
    """(min, max) angle in radians."""

    kp: float
    """PD position gain (Nm/rad)."""

    kd: float
    """PD damping gain (Nm·s/rad)."""


@dataclass
class Segment:
    """A rigid box segment. The root segment has joint=None."""

    size: Vec3
    """Half-extents of the box geom (MuJoCo convention)."""

    geom_pos: Vec3 = (0.0, 0.0, 0.0)
    """Center of the box geom in the segment's local frame."""

    mass: float = 0.5

    color: Vec4 = (0.85, 0.55, 0.20, 1.0)

    joint: Joint | None = None
    """Joint that attaches this segment to its parent. None only for root."""

    children: list[Segment] = field(default_factory=list)


@dataclass(frozen=True)
class SequenceStep:
    duration: float
    """Step duration in seconds."""

    targets: tuple[float, ...]
    """Target angle for each joint, in DFS order (excluding the root)."""


@dataclass(frozen=True)
class Sequence:
    cycle_duration: float
    """Total duration of one cycle (sum of step durations) in seconds."""

    steps: tuple[SequenceStep, ...]


@dataclass
class Genome:
    root: Segment

    sequence: Sequence

    initial_root_pos: Vec3 = (0.0, 0.0, 0.5)
    """Initial position of the root body in world coordinates."""

    id: str = "creature"

    version: int = 1

    fitness: dict[str, float] | None = None


def iter_joints(root: Segment) -> list[Joint]:
    """Return joints in DFS order, matching the convention used for sequence targets."""

    out: list[Joint] = []

    def _walk(seg: Segment) -> None:
        for child in seg.children:
            assert child.joint is not None, "Non-root segment must have a joint"
            out.append(child.joint)
            _walk(child)

    _walk(root)
    return out


def iter_segments(root: Segment) -> list[Segment]:
    """Return all segments in DFS order, including the root."""

    out: list[Segment] = []

    def _walk(seg: Segment) -> None:
        out.append(seg)
        for child in seg.children:
            _walk(child)

    _walk(root)
    return out


def n_joints(root: Segment) -> int:
    return len(iter_joints(root))
