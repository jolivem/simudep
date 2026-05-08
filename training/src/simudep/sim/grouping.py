"""Group population members by structural topology.

Same topology hash → identical MJCF body/joint/actuator counts → identical
MJX `Model` and `Data` pytree shapes → can be vmapped together. Each group
is then evaluated as a single batched MJX rollout (see `sim.rollout_mjx`),
and the JITted rollout function is cached per topology hash so that re-
encountering a topology in a later generation doesn't pay another compile.

The grouping is order-preserving in two senses:
  - groups are emitted in first-seen order;
  - within a group, individuals appear in their original input order.

`indices` lets the caller scatter the group's per-individual results back
into a single population-shaped output array.
"""

from __future__ import annotations

from dataclasses import dataclass

from simudep.genome.canonical import topology_hash
from simudep.genome.types import Genome


@dataclass(frozen=True)
class TopologyGroup:
    topo_hash: str
    """16-hex-char digest identifying the group (key for the JIT cache)."""

    indices: tuple[int, ...]
    """Indices of these genomes in the original input list."""

    genomes: tuple[Genome, ...]
    """Genomes belonging to this group, in `indices` order."""

    @property
    def size(self) -> int:
        return len(self.indices)


def group_by_topology(genomes: list[Genome]) -> list[TopologyGroup]:
    """Partition `genomes` by `topology_hash`. Returns groups in first-seen order."""

    order: list[str] = []
    buckets: dict[str, tuple[list[int], list[Genome]]] = {}
    for i, g in enumerate(genomes):
        h = topology_hash(g)
        if h not in buckets:
            buckets[h] = ([], [])
            order.append(h)
        buckets[h][0].append(i)
        buckets[h][1].append(g)
    return [
        TopologyGroup(topo_hash=h, indices=tuple(buckets[h][0]), genomes=tuple(buckets[h][1]))
        for h in order
    ]


__all__ = ["TopologyGroup", "group_by_topology"]
