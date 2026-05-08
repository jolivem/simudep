"""Hand-authored reference creatures used to validate the pipeline."""

from __future__ import annotations

import math

from simudep.genome.types import Genome, Joint, Segment, Sequence, SequenceStep


def tetrapod() -> Genome:
    """A simple 4-legged creature.

    Root: a flat box body lying horizontally.
    Legs: four vertical box segments, one at each corner of the body, each
    connected by a single revolute joint with axis along Y so legs swing
    forward/backward in the body's X-Z plane.

    Sequence: front legs and back legs swing in counter-phase to mimic a
    rough trot.
    """

    body_half = (0.18, 0.10, 0.04)
    leg_half = (0.025, 0.025, 0.10)
    body_mass = 1.2
    leg_mass = 0.15
    leg_color = (0.55, 0.35, 0.18, 1.0)

    def make_leg() -> Segment:
        return Segment(
            size=leg_half,
            geom_pos=(0.0, 0.0, -leg_half[2]),
            mass=leg_mass,
            color=leg_color,
        )

    def hip(anchor_x: float, anchor_y: float) -> Joint:
        return Joint(
            axis=(0.0, 1.0, 0.0),
            anchor_parent=(anchor_x, anchor_y, -body_half[2]),
            range=(-0.9, 0.9),
            kp=18.0,
            kd=0.8,
        )

    leg_fr = make_leg()
    leg_fr.joint = hip(+body_half[0] - 0.02, +body_half[1] - 0.01)

    leg_fl = make_leg()
    leg_fl.joint = hip(+body_half[0] - 0.02, -body_half[1] + 0.01)

    leg_br = make_leg()
    leg_br.joint = hip(-body_half[0] + 0.02, +body_half[1] - 0.01)

    leg_bl = make_leg()
    leg_bl.joint = hip(-body_half[0] + 0.02, -body_half[1] + 0.01)

    root = Segment(
        size=body_half,
        geom_pos=(0.0, 0.0, 0.0),
        mass=body_mass,
        color=(0.85, 0.55, 0.20, 1.0),
        children=[leg_fr, leg_fl, leg_br, leg_bl],
    )

    # DFS order of joints is [FR, FL, BR, BL].
    swing = math.radians(35.0)
    step_a = SequenceStep(duration=0.35, targets=(+swing, -swing, -swing, +swing))
    step_b = SequenceStep(duration=0.35, targets=(-swing, +swing, +swing, -swing))
    sequence = Sequence(cycle_duration=0.70, steps=(step_a, step_b))

    return Genome(
        root=root,
        sequence=sequence,
        initial_root_pos=(0.0, 0.0, body_half[2] + leg_half[2] * 2.0 + 0.005),
        id="tetrapod_ref",
    )
