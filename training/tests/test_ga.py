"""GA loop tests: small runs converge or at least don't diverge."""

from __future__ import annotations

from simudep.evo.ga import GAConfig, evolve
from simudep.genome.mutation import MutationRates


def test_tiny_ga_runs_to_completion() -> None:
    cfg = GAConfig(
        population_size=4,
        n_generations=2,
        rollout_duration=1.0,
        rollout_fps=30,
        seed=42,
        mutation_rates=MutationRates(),
    )
    pop = evolve(cfg)
    assert len(pop) == cfg.population_size
    # Sorted descending by score.
    scores = [ind.fitness["score"] for ind in pop]
    assert scores == sorted(scores, reverse=True)


def test_ga_seeded_runs_are_reproducible() -> None:
    cfg = GAConfig(
        population_size=4,
        n_generations=2,
        rollout_duration=1.0,
        rollout_fps=30,
        seed=123,
    )
    pop_a = evolve(cfg)
    pop_b = evolve(cfg)
    a_scores = [ind.fitness["score"] for ind in pop_a]
    b_scores = [ind.fitness["score"] for ind in pop_b]
    assert a_scores == b_scores
