"""Persist training data and selected individuals to disk.

Run layout:

    runs/<name>/
        config.json                        # GA hyperparams
        generations.jsonl                  # one line per generation: stats
        population/
            gen_0000/individuals.jsonl.gz  # full pop snapshot (gzipped JSONL)
            ...
        selected/
            <id>/                          # fresh deterministic rollouts
                genome.json
                trajectory.bin
                meta.json
            latest -> <id>                 # symlink to most recent

`update_latest_symlink(parent, name)` points `parent/latest` at `parent/name`,
giving the viz a stable URL to the most recently written individual.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from simudep.genome.types import Genome
from simudep.io.genome_json import genome_to_dict
from simudep.sim.rollout import RolloutResult


def write_selected(out_dir: Path | str, genome: Genome, result: RolloutResult) -> Path:
    """Write `genome.json`, `trajectory.bin`, `meta.json` to `out_dir`."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    genome_dict = genome_to_dict(genome)
    if result.fitness:
        genome_dict["fitness"] = result.fitness
    (out / "genome.json").write_text(json.dumps(genome_dict, indent=2))

    qpos = np.ascontiguousarray(result.qpos, dtype=np.float32)
    (out / "trajectory.bin").write_bytes(qpos.tobytes(order="C"))

    meta: dict[str, Any] = {
        "version": 1,
        "fps": result.fps,
        "dt": 1.0 / result.fps,
        "duration": result.duration,
        "timestep": result.timestep,
        "n_frames": int(qpos.shape[0]),
        "nq": result.nq,
        "n_joints": result.n_joints,
        "body_names": result.body_names,
        "fitness": result.fitness,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    return out


def update_latest_symlink(parent: Path | str, target_name: str) -> None:
    """Point `<parent>/latest` to `<parent>/<target_name>` (relative symlink)."""

    parent_path = Path(parent)
    parent_path.mkdir(parents=True, exist_ok=True)
    link_path = parent_path / "latest"
    if link_path.is_symlink() or link_path.exists():
        try:
            link_path.unlink()
        except IsADirectoryError:
            return
    os.symlink(target_name, link_path)


def write_run_config(run_dir: Path | str, config: Any) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(config)
    (run_path / "config.json").write_text(json.dumps(payload, indent=2))


def append_generation_stats(run_dir: Path | str, stats: Any) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_to_jsonable(stats))
    with (run_path / "generations.jsonl").open("a") as f:
        f.write(line + "\n")


def write_population_snapshot(
    run_dir: Path | str,
    generation: int,
    individuals: list[Any],
) -> Path:
    """Write a gzipped JSONL of {genome, fitness} for every individual."""

    run_path = Path(run_dir)
    snap_dir = run_path / "population" / f"gen_{generation:04d}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_file = snap_dir / "individuals.jsonl.gz"
    with gzip.open(out_file, "wt") as f:
        for ind in individuals:
            payload = {
                "genome": genome_to_dict(ind.genome),
                "fitness": ind.fitness,
            }
            f.write(json.dumps(payload) + "\n")
    return out_file


def read_population_snapshot(run_dir: Path | str, generation: int) -> list[dict[str, Any]]:
    """Load an entire generation's population from its gzipped JSONL snapshot."""

    snap = Path(run_dir) / "population" / f"gen_{generation:04d}" / "individuals.jsonl.gz"
    out: list[dict[str, Any]] = []
    with gzip.open(snap, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# -- internals ----------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of dataclasses / namedtuples to plain dict/list."""

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
