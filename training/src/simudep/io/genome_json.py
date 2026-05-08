"""Genome ↔ JSON serialization.

The schema is the authority for the Python ↔ Web contract; the TypeScript
side mirrors it in `viz/src/io/types.ts`. Trees are serialized recursively:
each non-root segment carries its `joint` describing how it attaches to its
parent.
"""

from __future__ import annotations

import json
from typing import Any

from simudep.genome.types import Genome, Joint, Segment, Sequence, SequenceStep


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    return {
        "version": genome.version,
        "id": genome.id,
        "initial_root_pos": list(genome.initial_root_pos),
        "root": _segment_to_dict(genome.root, is_root=True),
        "sequence": _sequence_to_dict(genome.sequence),
        "fitness": genome.fitness,
    }


def genome_from_dict(d: dict[str, Any]) -> Genome:
    return Genome(
        version=int(d.get("version", 1)),
        id=str(d.get("id", "creature")),
        initial_root_pos=_to_vec3(d["initial_root_pos"]),
        root=_segment_from_dict(d["root"], is_root=True),
        sequence=_sequence_from_dict(d["sequence"]),
        fitness=d.get("fitness"),
    )


def genome_dumps(genome: Genome, *, indent: int | None = 2) -> str:
    return json.dumps(genome_to_dict(genome), indent=indent)


def genome_loads(text: str) -> Genome:
    return genome_from_dict(json.loads(text))


# -- internals ----------------------------------------------------------


def _segment_to_dict(seg: Segment, *, is_root: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "size": list(seg.size),
        "geom_pos": list(seg.geom_pos),
        "mass": seg.mass,
        "color": list(seg.color),
        "children": [_segment_to_dict(c, is_root=False) for c in seg.children],
    }
    if not is_root:
        assert seg.joint is not None
        out["joint"] = _joint_to_dict(seg.joint)
    return out


def _segment_from_dict(d: dict[str, Any], *, is_root: bool) -> Segment:
    joint = None if is_root else _joint_from_dict(d["joint"])
    return Segment(
        size=_to_vec3(d["size"]),
        geom_pos=_to_vec3(d.get("geom_pos", [0.0, 0.0, 0.0])),
        mass=float(d["mass"]),
        color=_to_vec4(d["color"]),
        joint=joint,
        children=[_segment_from_dict(c, is_root=False) for c in d.get("children", [])],
    )


def _joint_to_dict(j: Joint) -> dict[str, Any]:
    return {
        "axis": list(j.axis),
        "anchor_parent": list(j.anchor_parent),
        "range": list(j.range),
        "kp": j.kp,
        "kd": j.kd,
    }


def _joint_from_dict(d: dict[str, Any]) -> Joint:
    rng = d["range"]
    return Joint(
        axis=_to_vec3(d["axis"]),
        anchor_parent=_to_vec3(d["anchor_parent"]),
        range=(float(rng[0]), float(rng[1])),
        kp=float(d["kp"]),
        kd=float(d["kd"]),
    )


def _sequence_to_dict(s: Sequence) -> dict[str, Any]:
    return {
        "cycle_duration": s.cycle_duration,
        "steps": [{"duration": st.duration, "targets": list(st.targets)} for st in s.steps],
    }


def _sequence_from_dict(d: dict[str, Any]) -> Sequence:
    steps = tuple(
        SequenceStep(duration=float(st["duration"]), targets=tuple(float(x) for x in st["targets"]))
        for st in d["steps"]
    )
    return Sequence(cycle_duration=float(d["cycle_duration"]), steps=steps)


def _to_vec3(x: Any) -> tuple[float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]))


def _to_vec4(x: Any) -> tuple[float, float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]))
