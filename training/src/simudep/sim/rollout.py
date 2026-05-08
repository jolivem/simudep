"""Single-creature CPU rollout (Phase 1).

GPU-batched MJX rollouts come in Phase 3. Phase 1 only needs to run one
creature at a time on the CPU to validate the full Genome → MJCF → simu →
trajectory pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from simudep.genome.types import Genome, Sequence
from simudep.mjcf.builder import build_mjcf


@dataclass
class RolloutResult:
    qpos: np.ndarray
    """(n_frames, nq) float32 array of generalized positions, sampled at `fps`."""

    fps: int
    duration: float
    timestep: float
    body_names: list[str]
    nq: int
    n_joints: int
    fitness: dict[str, float]
    """{distance, energy, score} where score = distance - alpha * energy."""


def rollout_cpu(
    genome: Genome,
    *,
    duration: float = 5.0,
    fps: int = 60,
    timestep: float = 0.005,
    energy_alpha: float = 1e-3,
) -> RolloutResult:
    """Simulate the genome on the CPU and return the qpos trajectory + fitness."""

    xml = build_mjcf(genome, timestep=timestep)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    n_frames = int(round(duration * fps))
    steps_per_frame = max(1, round(1.0 / (fps * timestep)))

    qpos_log = np.empty((n_frames, model.nq), dtype=np.float32)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
        for i in range(model.nbody)
    ]

    initial_xy = np.array(genome.initial_root_pos[:2], dtype=np.float64)
    energy_acc = 0.0

    for frame_idx in range(n_frames):
        qpos_log[frame_idx] = data.qpos.astype(np.float32)
        for _ in range(steps_per_frame):
            targets = _ctrl_at(float(data.time), genome.sequence)
            data.ctrl[:] = targets
            mujoco.mj_step(model, data)
            energy_acc += float(np.sum(np.square(data.ctrl))) * timestep

    final_xy = data.qpos[:2].astype(np.float64)
    distance = float(np.linalg.norm(final_xy - initial_xy))
    score = distance - energy_alpha * energy_acc

    return RolloutResult(
        qpos=qpos_log,
        fps=fps,
        duration=duration,
        timestep=timestep,
        body_names=body_names,
        nq=int(model.nq),
        n_joints=int(model.nu),
        fitness={"distance": distance, "energy": energy_acc, "score": score},
    )


def _ctrl_at(t: float, sequence: Sequence) -> tuple[float, ...]:
    """Return the joint-angle targets active at time `t` (cyclic)."""

    cycle_t = t % sequence.cycle_duration
    acc = 0.0
    for step in sequence.steps:
        acc += step.duration
        if cycle_t < acc:
            return step.targets
    return sequence.steps[-1].targets
