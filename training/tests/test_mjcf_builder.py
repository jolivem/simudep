"""MJCF builder tests: produces a valid MuJoCo model with the expected shape."""

from __future__ import annotations

import mujoco

from simudep.genome.builtin import tetrapod
from simudep.genome.types import iter_joints
from simudep.mjcf.builder import build_mjcf


def test_tetrapod_mjcf_parses_without_warnings() -> None:
    g = tetrapod()
    xml = build_mjcf(g)
    # If MuJoCo can't parse it, this raises.
    model = mujoco.MjModel.from_xml_string(xml)

    n_joints = len(iter_joints(g.root))
    # Free joint contributes 7 qpos; each hinge contributes 1.
    assert model.nq == 7 + n_joints
    assert model.nu == n_joints
    # world + root body + n_joints children.
    assert model.nbody == 2 + n_joints
    # 1 freejoint + n hinges.
    assert model.njnt == 1 + n_joints


def test_actuator_names_match_joint_dfs_order() -> None:
    g = tetrapod()
    model = mujoco.MjModel.from_xml_string(build_mjcf(g))
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        assert actuator_name == f"a{i}"
        assert joint_name == f"j{i}"
