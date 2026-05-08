"""Topology hashing.

Two genomes belong to the same topology *group* when their tree structures
(number of children at each node, in DFS order) are identical. Continuous
parameters (sizes, masses, kp/kd, joint axes, anchors, sequence targets)
are deliberately excluded — they vary per individual within a group, but
none of them changes MJCF body/joint/actuator counts and so none of them
changes the MJX `Model`/`Data` pytree shapes that we vmap over.

The signature is a parens-encoded DFS encoding of the tree, e.g.
"(()()()())" for a creature whose root has four leaf children. The hash
is `sha256(signature)[:16]` for a fixed-length identifier; `signature`
itself is exposed for debugging and human inspection.
"""

from __future__ import annotations

import hashlib

from simudep.genome.types import Genome, Segment


def topology_signature(genome: Genome) -> str:
    """Order-sensitive parens encoding of the genome's tree structure."""

    return _signature(genome.root)


def topology_hash(genome: Genome) -> str:
    """Short hex digest of `topology_signature(genome)`."""

    sig = topology_signature(genome)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def _signature(seg: Segment) -> str:
    inner = "".join(_signature(c) for c in seg.children)
    return "(" + inner + ")"


__all__ = ["topology_hash", "topology_signature"]
