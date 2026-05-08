"""simudep CLI entry point.

Phase 2 surface:
    simudep doctor         Check the runtime (JAX/CUDA, MuJoCo).
    simudep inspect-one    Simulate one creature and write a run dir.
    simudep train          Run a GA (CPU, naive) and persist generations.
    simudep replay         Re-simulate one individual from a training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simudep import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simudep", description="Evolutionary creature simulator.")
    parser.add_argument("--version", action="version", version=f"simudep {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("doctor", help="Check the runtime environment (JAX/CUDA, MuJoCo).")

    p_bench = sub.add_parser(
        "bench",
        help=(
            "Compare CPU sequential rollout vs MJX batched rollout on a random "
            "population. Phase-3 acceptance: speedup ≥10× at pop=128."
        ),
    )
    p_bench.add_argument("--pop", type=int, default=128)
    p_bench.add_argument("--duration", type=float, default=5.0)
    p_bench.add_argument("--fps", type=int, default=60)
    p_bench.add_argument("--timestep", type=float, default=0.005)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument(
        "--n-segments",
        type=int,
        default=None,
        help="Force a fixed segment count (default: random per individual).",
    )

    p_inspect = sub.add_parser(
        "inspect-one",
        help="Simulate one creature and write a run dir under runs/.",
    )
    p_inspect.add_argument(
        "--kind",
        choices=("random", "tetrapod"),
        default="random",
        help="Source of the genome (default: random).",
    )
    p_inspect.add_argument("--out", type=Path, default=Path("runs/inspect"))
    p_inspect.add_argument("--seed", type=int, default=None)
    p_inspect.add_argument("--n-segments", type=int, default=None)
    p_inspect.add_argument("--duration", type=float, default=5.0)
    p_inspect.add_argument("--fps", type=int, default=60)
    p_inspect.add_argument("--timestep", type=float, default=0.005)

    p_train = sub.add_parser(
        "train",
        help="Run a GA on CPU and persist per-generation snapshots + the best individual.",
    )
    p_train.add_argument("--out", type=Path, required=True)
    p_train.add_argument("--pop", type=int, default=32, dest="population_size")
    p_train.add_argument("--gens", type=int, default=30, dest="n_generations")
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--duration", type=float, default=5.0)
    p_train.add_argument("--fps", type=int, default=60)
    p_train.add_argument("--timestep", type=float, default=0.005)
    p_train.add_argument("--energy-alpha", type=float, default=1e-3)
    p_train.add_argument("--p-crossover", type=float, default=0.5)
    p_train.add_argument("--tournament-size", type=int, default=3)
    p_train.add_argument("--elite-frac", type=float, default=0.1)
    p_train.add_argument(
        "--mjx",
        action="store_true",
        help=(
            "Evaluate offspring with the batched MJX rollout (GPU). Requires CUDA. "
            "See sim.rollout_mjx for the per-group canonical-model caveat."
        ),
    )

    p_replay = sub.add_parser(
        "replay",
        help="Re-simulate one individual from a training run and write a viewable selected dir.",
    )
    p_replay.add_argument("--run", type=Path, required=True)
    p_replay.add_argument("--gen", type=int, default=-1, help="Generation index (-1 = last).")
    p_replay.add_argument("--rank", type=int, default=0, help="Rank in the generation (0 = best).")
    p_replay.add_argument("--duration", type=float, default=5.0)
    p_replay.add_argument("--fps", type=int, default=60)
    p_replay.add_argument("--timestep", type=float, default=0.005)
    p_replay.add_argument("--energy-alpha", type=float, default=1e-3)

    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "bench":
        return _bench(args)
    if args.cmd == "inspect-one":
        return _inspect_one(
            kind=args.kind,
            out_root=args.out,
            seed=args.seed,
            n_segments=args.n_segments,
            duration=args.duration,
            fps=args.fps,
            timestep=args.timestep,
        )
    if args.cmd == "train":
        return _train(args)
    if args.cmd == "replay":
        return _replay(args)

    parser.print_help()
    return 0


def _doctor() -> int:
    import jax
    import mujoco

    print(f"simudep {__version__}")
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"mujoco {mujoco.__version__}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    """Time CPU-sequential vs MJX-batched rollouts on a random population."""

    import time

    import numpy as np

    from simudep.genome.random_init import random_genome
    from simudep.sim.grouping import group_by_topology
    from simudep.sim.rollout import rollout_cpu
    from simudep.sim.rollout_mjx import clear_jit_cache, rollout_mjx_population

    rng = np.random.default_rng(args.seed)
    pop = [
        random_genome(rng, n_segments=args.n_segments, name=f"bench{i:04d}")
        for i in range(args.pop)
    ]
    groups = group_by_topology(pop)
    print(
        f"Bench: pop={args.pop}  duration={args.duration}s  fps={args.fps}  "
        f"timestep={args.timestep}s  seed={args.seed}",
    )
    print(
        f"  unique topologies: {len(groups)}  "
        f"top group sizes: {sorted([g.size for g in groups], reverse=True)[:5]}",
    )

    print("CPU sequential rollout...")
    t0 = time.perf_counter()
    for g in pop:
        rollout_cpu(
            g,
            duration=args.duration,
            fps=args.fps,
            timestep=args.timestep,
        )
    t_cpu = time.perf_counter() - t0
    print(f"  CPU total: {t_cpu:.2f}s   ({t_cpu / args.pop * 1000:.1f} ms / individual)")

    print("MJX batched rollout (warmup, includes JIT compile)...")
    clear_jit_cache()
    # Warmup with a tiny prefix so the timed run has hot caches.
    t0 = time.perf_counter()
    rollout_mjx_population(
        pop[: min(8, args.pop)],
        duration=args.duration,
        fps=args.fps,
        timestep=args.timestep,
    )
    t_warm = time.perf_counter() - t0
    print(f"  warmup: {t_warm:.2f}s")

    print("MJX batched rollout (timed run)...")
    t0 = time.perf_counter()
    rollout_mjx_population(
        pop,
        duration=args.duration,
        fps=args.fps,
        timestep=args.timestep,
    )
    t_mjx = time.perf_counter() - t0
    print(f"  MJX total: {t_mjx:.2f}s   ({t_mjx / args.pop * 1000:.1f} ms / individual)")

    speedup = t_cpu / max(t_mjx, 1e-9)
    target = 10.0
    verdict = "PASS" if speedup >= target else "FAIL"
    print(f"\n>>> Speedup: {speedup:.1f}×  (target ≥{target:.0f}×)  [{verdict}]")
    return 0 if speedup >= target else 1


def _inspect_one(
    *,
    kind: str,
    out_root: Path,
    seed: int | None,
    n_segments: int | None,
    duration: float,
    fps: int,
    timestep: float,
) -> int:
    import numpy as np

    from simudep.io.run_writer import update_latest_symlink, write_selected
    from simudep.sim.rollout import rollout_cpu

    if kind == "tetrapod":
        from simudep.genome.builtin import tetrapod

        genome = tetrapod()
    else:
        from simudep.genome.random_init import random_genome

        rng = np.random.default_rng(seed)
        genome = random_genome(rng, n_segments=n_segments)

    n_seg_actual = sum(1 for _ in _walk_segments(genome.root))
    n_joints_actual = sum(1 for _ in _walk_joints(genome.root))
    print(
        f"Simulating '{genome.id}' "
        f"({n_seg_actual} segments, {n_joints_actual} joints) "
        f"for {duration}s at {fps} fps (timestep={timestep}s)...",
    )
    result = rollout_cpu(genome, duration=duration, fps=fps, timestep=timestep)
    target = out_root / "selected" / genome.id
    write_selected(target, genome, result)
    update_latest_symlink(out_root / "selected", genome.id)

    f = result.fitness
    print(f"  distance = {f['distance']:.4f} m")
    print(f"  energy   = {f['energy']:.4f}")
    print(f"  score    = {f['score']:.4f}")
    print(f"Wrote: {target}")
    return 0


def _walk_segments(root):
    yield root
    for child in root.children:
        yield from _walk_segments(child)


def _walk_joints(root):
    for child in root.children:
        yield child.joint
        yield from _walk_joints(child)


def _train(args: argparse.Namespace) -> int:
    import time

    from simudep.evo.ga import GAConfig, GenerationStats, Individual, evolve
    from simudep.io.run_writer import (
        append_generation_stats,
        update_latest_symlink,
        write_population_snapshot,
        write_run_config,
        write_selected,
    )
    from simudep.sim.rollout import rollout_cpu

    cfg = GAConfig(
        population_size=args.population_size,
        n_generations=args.n_generations,
        tournament_size=args.tournament_size,
        elite_frac=args.elite_frac,
        p_crossover=args.p_crossover,
        rollout_duration=args.duration,
        rollout_fps=args.fps,
        rollout_timestep=args.timestep,
        energy_alpha=args.energy_alpha,
        seed=args.seed,
        use_mjx=args.mjx,
    )

    run_dir: Path = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir, cfg)

    start = time.perf_counter()
    last_pop: list[Individual] = []

    def _on_gen(stats: GenerationStats, pop: list[Individual]) -> None:
        nonlocal last_pop
        last_pop = pop
        append_generation_stats(run_dir, stats)
        write_population_snapshot(run_dir, stats.generation, pop)
        elapsed = time.perf_counter() - start
        print(
            f"  gen {stats.generation:3d}  "
            f"score max={stats.score_max:7.3f}  mean={stats.score_mean:7.3f}  "
            f"dist max={stats.distance_max:6.3f}  "
            f"segs={stats.n_segments_mean:4.1f}  "
            f"({elapsed:5.1f}s)",
        )

    print(
        f"Training: pop={cfg.population_size}, gens={cfg.n_generations}, "
        f"duration={cfg.rollout_duration}s/individual, seed={cfg.seed} → {run_dir}",
    )
    final = evolve(cfg, on_generation=_on_gen)

    # Re-run the best individual deterministically and write it for the viz.
    best = final[0]
    print(f"Best of last gen: {best.genome.id}  score={best.fitness['score']:.3f}")
    result = rollout_cpu(
        best.genome,
        duration=cfg.rollout_duration,
        fps=cfg.rollout_fps,
        timestep=cfg.rollout_timestep,
        energy_alpha=cfg.energy_alpha,
    )
    target = run_dir / "selected" / best.genome.id
    write_selected(target, best.genome, result)
    update_latest_symlink(run_dir / "selected", best.genome.id)
    print(f"Wrote: {target}")
    print(f"Tip: open the viz with ?run=/runs/{run_dir.name}/selected/latest")
    _ = last_pop  # kept around for debugging
    return 0


def _replay(args: argparse.Namespace) -> int:
    from simudep.io.genome_json import genome_from_dict
    from simudep.io.run_writer import (
        read_population_snapshot,
        update_latest_symlink,
        write_selected,
    )
    from simudep.sim.rollout import rollout_cpu

    run_dir: Path = args.run
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 2

    gen = args.gen
    if gen < 0:
        # Find the last available generation.
        pop_dir = run_dir / "population"
        gen_dirs = sorted(pop_dir.glob("gen_*"))
        if not gen_dirs:
            print(f"error: no generations under {pop_dir}", file=sys.stderr)
            return 2
        gen = int(gen_dirs[-1].name.removeprefix("gen_"))

    snap = read_population_snapshot(run_dir, gen)
    if not snap:
        print(f"error: empty population for generation {gen}", file=sys.stderr)
        return 2

    snap.sort(key=lambda r: r["fitness"]["score"], reverse=True)
    if args.rank >= len(snap):
        print(f"error: rank {args.rank} out of range (pop size = {len(snap)})", file=sys.stderr)
        return 2

    record = snap[args.rank]
    genome = genome_from_dict(record["genome"])

    print(
        f"Replaying gen={gen} rank={args.rank} id={genome.id} "
        f"(stored score={record['fitness']['score']:.3f}) ...",
    )
    result = rollout_cpu(
        genome,
        duration=args.duration,
        fps=args.fps,
        timestep=args.timestep,
        energy_alpha=args.energy_alpha,
    )
    target = run_dir / "selected" / genome.id
    write_selected(target, genome, result)
    update_latest_symlink(run_dir / "selected", genome.id)
    print(
        f"  fresh score = {result.fitness['score']:.3f}  "
        f"distance = {result.fitness['distance']:.3f} m",
    )
    print(f"Wrote: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
