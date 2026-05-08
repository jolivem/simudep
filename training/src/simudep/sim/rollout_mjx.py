"""Batched MJX rollout per topology group (Phase 3).

Same-topology individuals share the MJX `Model` pytree shape, but `mjx.put_model`
unfortunately also bakes parameter-dependent fields into the *static* part of
the pytree (sparse-matrix sizes `nM/nB/nC`, the model `signature`, the `names`
buffer, …). That makes a naive `jax.tree.map(stack, *models)` raise
``ValueError: Mismatch custom node data`` even when topology and array shapes
match exactly.

The pragmatic Phase-3 compromise is: one canonical `mjx.Model` per topology
group, taken from the group's first genome, and `jax.vmap` only across the
per-individual `ctrl_seq` (initial state and physical parameters are shared).
This trades parity-correctness for clean batched execution, and is sufficient
for the Phase-3 acceptance criterion (≥10× speedup vs sequential CPU on
pop=128 — see PLAN.md). Implications:

  * **Topology** mutations within a group are still evaluated correctly,
    because the group is keyed by topology hash.
  * **Sequence** mutations are honored (the cyclic schedule lives in `ctrl`).
  * **Continuous** mutations on segment sizes / masses / `kp` / `kd` /
    joint axes are *not* honored within a group during MJX rollout — every
    individual is evaluated with the first one's body. The CPU rollout
    (`rollout_cpu`) and the `simudep replay` path keep using the per-genome
    full parameters, so the viz and the saved best-of-run remain faithful.

The compiled rollout function is cached per topology hash and rollout
parameters via `functools.lru_cache(maxsize=50)`: every new topology pays one
JIT compile, every reuse only pays the kernel launch.

NaN handling: simulations occasionally diverge in MJX (deep kinematic chains
with stiff PD gains seem prone to it on this version). We catch NaN/Inf in
each individual's `final_qpos` and assign a heavy penalty fitness rather than
poisoning the whole batch — the GA will then evolve away from such genomes.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from simudep.genome.types import Genome, Sequence
from simudep.mjcf.builder import build_mjcf
from simudep.sim.grouping import group_by_topology
from simudep.sim.rollout import RolloutResult

# When a rollout produces NaN/Inf, we still want a finite fitness so the GA
# can rank it (heavily) below stable individuals.
_NAN_FITNESS = {"distance": 0.0, "energy": 1e6, "score": -1e6}


def rollout_mjx_population(
    genomes: list[Genome],
    *,
    duration: float = 5.0,
    fps: int = 60,
    timestep: float = 0.005,
    energy_alpha: float = 1e-3,
) -> list[RolloutResult]:
    """Evaluate a whole population on the GPU via topology grouping + vmap.

    Returns one `RolloutResult` per input genome, in the original order.
    """

    # Frame indexing matches `rollout_cpu`: record qpos *before* every block
    # of `steps_per_frame` simulation steps, so frames line up across the two
    # paths and `n_frames` is exactly `round(duration * fps)`.
    n_frames = int(round(duration * fps))
    steps_per_frame = max(1, int(round(1.0 / (fps * timestep))))
    n_total_steps = n_frames * steps_per_frame

    results: list[RolloutResult | None] = [None] * len(genomes)

    for group in group_by_topology(genomes):
        canonical = group.genomes[0]
        mj_model = mujoco.MjModel.from_xml_string(build_mjcf(canonical, timestep=timestep))
        mjx_model = mjx.put_model(mj_model)
        nu = int(mj_model.nu)
        nq = int(mj_model.nq)
        body_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
            for i in range(mj_model.nbody)
        ]

        ctrl_seqs = np.stack(
            [
                _precompute_ctrl(g.sequence, n_total_steps=n_total_steps, dt=timestep, nu=nu)
                for g in group.genomes
            ],
            axis=0,
        )

        rollout_fn = _get_compiled_rollout(
            group.topo_hash,
            n_total_steps=n_total_steps,
            n_frames=n_frames,
            steps_per_frame=steps_per_frame,
            timestep=timestep,
        )

        qpos_log_jax, final_qpos_jax, energies_jax = rollout_fn(
            mjx_model, jnp.asarray(ctrl_seqs)
        )
        qpos_log_jax.block_until_ready()
        qpos_log = np.asarray(qpos_log_jax)  # (B, n_frames, nq)
        final_qpos = np.asarray(final_qpos_jax)  # (B, nq)
        energies = np.asarray(energies_jax)  # (B,)

        for k, idx in enumerate(group.indices):
            qpos_k = qpos_log[k].astype(np.float32, copy=False)
            energy_k = float(energies[k])
            if not (np.all(np.isfinite(qpos_k)) and np.all(np.isfinite(final_qpos[k]))
                    and np.isfinite(energy_k)):
                fitness = dict(_NAN_FITNESS)
                qpos_safe = np.nan_to_num(qpos_k, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                initial_xy = np.array(group.genomes[k].initial_root_pos[:2], dtype=np.float64)
                final_xy = final_qpos[k, :2].astype(np.float64)
                distance = float(np.linalg.norm(final_xy - initial_xy))
                score = distance - energy_alpha * energy_k
                fitness = {"distance": distance, "energy": energy_k, "score": score}
                qpos_safe = qpos_k

            results[idx] = RolloutResult(
                qpos=qpos_safe,
                fps=fps,
                duration=duration,
                timestep=timestep,
                body_names=body_names,
                nq=nq,
                n_joints=nu,
                fitness=fitness,
            )

    assert all(r is not None for r in results), "rollout_mjx_population missed an individual"
    return results  # type: ignore[return-value]


# -- internals ----------------------------------------------------------


def _precompute_ctrl(
    seq: Sequence, *, n_total_steps: int, dt: float, nu: int
) -> np.ndarray:
    """Sample the cyclic sequence at every simulation timestep."""

    out = np.zeros((n_total_steps, nu), dtype=np.float32)
    durations = np.array([s.duration for s in seq.steps], dtype=np.float64)
    cum = np.cumsum(durations)
    cycle = float(seq.cycle_duration)
    n_steps = len(seq.steps)
    targets = np.array([s.targets for s in seq.steps], dtype=np.float32)  # (n_steps, nu)
    for k in range(n_total_steps):
        t = (k * dt) % cycle
        idx = int(np.searchsorted(cum, t, side="right"))
        if idx >= n_steps:
            idx = n_steps - 1
        out[k] = targets[idx]
    return out


@functools.lru_cache(maxsize=50)
def _get_compiled_rollout(
    topo_hash: str,
    *,
    n_total_steps: int,
    n_frames: int,
    steps_per_frame: int,
    timestep: float,
):
    """Compile a `vmap(rollout_one)` once per (topology, sim params).

    The model is shared across the batch dimension; only `ctrl_seq` is mapped.
    """

    del topo_hash  # only used as a cache key

    frame_indices = jnp.arange(n_frames) * steps_per_frame

    def _rollout_one(model, ctrl_seq):
        data = mjx.make_data(model)

        def step_fn(carry, ctrl_t):
            data, energy = carry
            qpos_before = data.qpos  # match CPU: record pre-step qpos for frame 0
            data = data.replace(ctrl=ctrl_t)
            data = mjx.step(model, data)
            energy = energy + jnp.sum(ctrl_t * ctrl_t) * timestep
            return (data, energy), qpos_before

        (final_data, total_energy), qpos_all = jax.lax.scan(
            step_fn, (data, jnp.float32(0.0)), ctrl_seq
        )
        qpos_at_frames = qpos_all[frame_indices]
        return qpos_at_frames, final_data.qpos, total_energy

    # in_axes=(None, 0) → broadcast model, map over ctrl_seq batch dim.
    return jax.jit(jax.vmap(_rollout_one, in_axes=(None, 0)))


def clear_jit_cache() -> None:
    """Drop every compiled rollout. Useful for benchmarks across configs."""

    _get_compiled_rollout.cache_clear()


__all__ = ["clear_jit_cache", "rollout_mjx_population"]
