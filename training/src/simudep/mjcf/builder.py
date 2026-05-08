"""Genome → MJCF XML builder.

Produces a complete, self-contained MuJoCo model with the ground plane and
PD-controlled actuators on every joint. Joints and bodies are named in DFS
order so that MuJoCo's `qpos` and `ctrl` arrays line up with the sequence
target ordering used elsewhere in the codebase:

    qpos layout: [root_pos(3), root_quat(4), j0, j1, ..., j_{N-1}]
    ctrl layout: [j0, j1, ..., j_{N-1}]

Bodies are named "b{idx}" and joints "j{idx}" with idx = DFS index.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from simudep.genome.types import Genome, Segment, Vec3, Vec4


def build_mjcf(genome: Genome, *, timestep: float = 0.005) -> str:
    """Serialize the genome to a MuJoCo MJCF XML string."""

    root = ET.Element("mujoco", model=genome.id)

    ET.SubElement(
        root,
        "option",
        timestep=f"{timestep}",
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )

    # Disable self-collisions per the project plan (MVP). The standard
    # MuJoCo trick: split bodies and ground into mutually-affinitive groups
    # so body↔body has no overlap in (contype & conaffinity).
    #
    #   ground:    contype=1, conaffinity=2
    #   body seg:  contype=2, conaffinity=1   (via the "seg" default class)
    #
    # ground↔seg → (1&1)|(2&2) = 3 → collide
    # seg↔seg   → (2&1)|(2&1) = 0 → no collide ✓
    defaults = ET.SubElement(root, "default")
    ET.SubElement(
        defaults,
        "geom",
        contype="1",
        conaffinity="2",
        condim="3",
        friction="0.9 0.05 0.05",
    )
    ET.SubElement(defaults, "joint", armature="0.001", damping="0.05")
    seg_class = ET.SubElement(defaults, "default", {"class": "seg"})
    ET.SubElement(seg_class, "geom", contype="2", conaffinity="1")

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 5",
        dir="0 0 -1",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        size="50 50 0.1",
        rgba="0.30 0.32 0.34 1",
    )

    body_counter = _Counter()
    joint_counter = _Counter()

    root_body = ET.SubElement(
        worldbody,
        "body",
        name=f"b{body_counter.next()}",
        pos=_v(genome.initial_root_pos),
    )
    ET.SubElement(root_body, "freejoint")
    _emit_segment(root_body, genome.root, body_counter, joint_counter, is_root=True)

    actuator = ET.SubElement(root, "actuator")
    _emit_actuators(genome.root, actuator)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


# -- internals ----------------------------------------------------------


def _emit_segment(
    parent_xml: ET.Element,
    seg: Segment,
    body_counter: _Counter,
    joint_counter: _Counter,
    *,
    is_root: bool,
) -> None:
    ET.SubElement(
        parent_xml,
        "geom",
        {"class": "seg"},
        type="box",
        size=_v(seg.size),
        pos=_v(seg.geom_pos),
        rgba=_v4(seg.color),
        mass=f"{seg.mass}",
    )

    for child in seg.children:
        assert child.joint is not None, "Non-root segment must have a joint"
        body_idx = body_counter.next()
        joint_idx = joint_counter.next()
        body_xml = ET.SubElement(
            parent_xml,
            "body",
            name=f"b{body_idx}",
            pos=_v(child.joint.anchor_parent),
        )
        ET.SubElement(
            body_xml,
            "joint",
            name=f"j{joint_idx}",
            type="hinge",
            axis=_v(child.joint.axis),
            range=f"{child.joint.range[0]} {child.joint.range[1]}",
        )
        _emit_segment(body_xml, child, body_counter, joint_counter, is_root=False)

    _ = is_root


def _emit_actuators(root: Segment, actuator_xml: ET.Element) -> None:
    """Emit one PD-style general actuator per joint, matching DFS order."""

    counter = _Counter()

    def _walk(seg: Segment) -> None:
        for child in seg.children:
            assert child.joint is not None
            idx = counter.next()
            kp = child.joint.kp
            kd = child.joint.kd
            ET.SubElement(
                actuator_xml,
                "general",
                name=f"a{idx}",
                joint=f"j{idx}",
                ctrlrange=f"{child.joint.range[0]} {child.joint.range[1]}",
                gaintype="fixed",
                gainprm=f"{kp} 0 0",
                biastype="affine",
                biasprm=f"0 {-kp} {-kd}",
            )
            _walk(child)

    _walk(root)


def _v(t: Vec3) -> str:
    return f"{t[0]} {t[1]} {t[2]}"


def _v4(t: Vec4) -> str:
    return f"{t[0]} {t[1]} {t[2]} {t[3]}"


class _Counter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        i = self._n
        self._n += 1
        return i
