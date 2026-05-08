"""Phase 2 genetic algorithm.

Naive single-process loop:

    1. Initialize a population of random genomes.
    2. For each generation:
         - Evaluate every individual via `rollout_cpu` (CPU, sequential).
         - Carry the top fraction over as "elites" (deep-copied so parents
           don't share state with offspring).
         - Fill the rest of the new population by repeating:
             * tournament-pick two parents
             * with prob `p_crossover`, splice one of B's sub-trees onto a
               copy of A; otherwise just deep-copy A
             * always mutate the offspring.

GPU batching, CMA-ES on the sequence and topology grouping all come in
later phases. This module only depends on mujoco (CPU) and numpy.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from simudep.genome.crossover import crossover
from simudep.genome.mutation import MutationRates, mutate
from simudep.genome.random_init import random_genome
from simudep.genome.types import Genome
from simudep.sim.rollout import RolloutResult, rollout_cpu


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 32
    n_generations: int = 30
    tournament_size: int = 3
    elite_frac: float = 0.1
    p_crossover: float = 0.5
    mutation_rates: MutationRates = field(default_factory=MutationRates)
    rollout_duration: float = 5.0
    rollout_fps: int = 60
    rollout_timestep: float = 0.005
    energy_alpha: float = 1e-3
    seed: int | None = None
    use_mjx: bool = False
    """When True, evaluate offspring with `rollout_mjx_population` (batched
    GPU). Elites carry their stored fitness over from the previous generation
    so we never re-evaluate them. See `sim.rollout_mjx` for the per-group
    canonical-model caveat."""


@dataclass
class Individual:
    genome: Genome
    fitness: dict[str, float]


@dataclass
class GenerationStats:
    generation: int
    score_max: float
    score_mean: float
    score_min: float
    distance_max: float
    distance_mean: float
    energy_mean: float
    n_segments_mean: float
    best_id: str


def evaluate(genome: Genome, cfg: GAConfig) -> tuple[Individual, RolloutResult]:
    result = rollout_cpu(
        genome,
        duration=cfg.rollout_duration,
        fps=cfg.rollout_fps,
        timestep=cfg.rollout_timestep,
        energy_alpha=cfg.energy_alpha,
    )
    return Individual(genome=genome, fitness=result.fitness), result


def evaluate_population(genomes: list[Genome], cfg: GAConfig) -> list[Individual]:
    """Score a list of genomes via the configured backend (CPU or MJX-batched).

    The MJX path imports lazily so CPU-only runs don't pay for the JAX/CUDA
    initialization cost.
    """

    if not genomes:
        return []
    if cfg.use_mjx:
        from simudep.sim.rollout_mjx import rollout_mjx_population

        results = rollout_mjx_population(
            genomes,
            duration=cfg.rollout_duration,
            fps=cfg.rollout_fps,
            timestep=cfg.rollout_timestep,
            energy_alpha=cfg.energy_alpha,
        )
        return [
            Individual(genome=g, fitness=r.fitness)
            for g, r in zip(genomes, results, strict=True)
        ]
    return [evaluate(g, cfg)[0] for g in genomes]


def evolve(
    cfg: GAConfig,
    *,
    on_generation: Callable[[GenerationStats, list[Individual]], None] | None = None,
) -> list[Individual]:
    """Run the GA and return the final population sorted by fitness desc."""

    rng = np.random.default_rng(cfg.seed)

    # Generation 0: random initialization.
    gen0_genomes = [random_genome(rng) for _ in range(cfg.population_size)]
    population = evaluate_population(gen0_genomes, cfg)
    population.sort(key=lambda i: i.fitness["score"], reverse=True)
    if on_generation is not None:
        on_generation(_stats(0, population), population)

    n_elites = max(1, int(round(cfg.elite_frac * cfg.population_size)))

    for gen in range(1, cfg.n_generations):
        elites = [
            Individual(copy.deepcopy(e.genome), dict(e.fitness))
            for e in population[:n_elites]
        ]

        # Build all offspring genomes first, then evaluate them in a single batch.
        offspring_genomes: list[Genome] = []
        while len(elites) + len(offspring_genomes) < cfg.population_size:
            a = _tournament(population, cfg.tournament_size, rng)
            if rng.random() < cfg.p_crossover and cfg.population_size > 1:
                b = _tournament(population, cfg.tournament_size, rng)
                child_genome = crossover(a.genome, b.genome, rng)
            else:
                child_genome = copy.deepcopy(a.genome)
            child_genome = mutate(child_genome, rng, cfg.mutation_rates)
            child_genome.id = _new_id(rng, gen)
            offspring_genomes.append(child_genome)

        offspring = evaluate_population(offspring_genomes, cfg)
        next_pop = elites + offspring

        next_pop.sort(key=lambda i: i.fitness["score"], reverse=True)
        population = next_pop
        if on_generation is not None:
            on_generation(_stats(gen, population), population)

    return population


# -- helpers ------------------------------------------------------------


def _tournament(pop: list[Individual], k: int, rng: np.random.Generator) -> Individual:
    idxs = rng.integers(0, len(pop), size=min(k, len(pop)))
    candidates = [pop[int(i)] for i in idxs]
    candidates.sort(key=lambda i: i.fitness["score"], reverse=True)
    return candidates[0]


def _stats(gen: int, sorted_pop: list[Individual]) -> GenerationStats:
    scores = np.array([i.fitness["score"] for i in sorted_pop], dtype=np.float64)
    distances = np.array([i.fitness["distance"] for i in sorted_pop], dtype=np.float64)
    energies = np.array([i.fitness["energy"] for i in sorted_pop], dtype=np.float64)
    n_segs = np.array(
        [_count_segments(i.genome) for i in sorted_pop], dtype=np.float64
    )
    return GenerationStats(
        generation=gen,
        score_max=float(scores.max()),
        score_mean=float(scores.mean()),
        score_min=float(scores.min()),
        distance_max=float(distances.max()),
        distance_mean=float(distances.mean()),
        energy_mean=float(energies.mean()),
        n_segments_mean=float(n_segs.mean()),
        best_id=sorted_pop[0].genome.id,
    )


def _count_segments(g: Genome) -> int:
    n = 0
    stack = [g.root]
    while stack:
        s = stack.pop()
        n += 1
        stack.extend(s.children)
    return n


def _new_id(rng: np.random.Generator, gen: int) -> str:
    return f"g{gen:04d}_{int(rng.integers(0, 1 << 24)):06x}"


__all__ = [
    "GAConfig",
    "GenerationStats",
    "Individual",
    "evaluate_population",
    "evolve",
]
