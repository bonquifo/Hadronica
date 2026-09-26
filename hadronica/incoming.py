"""Incoming state chosen by the user: two particles, two momenta, one angle.

The angle is the angle between the two momentum vectors. 180° is head-on.
Speed is not an independent control: β = |p| / E once the mass and momentum are fixed.
Only initial states that already have a Born formula are turned into products.
"""

from __future__ import annotations

import math

from hadronica.generator import Event, ParticleRecord, _assert_conserved
from hadronica.lorentz import FourVector, boost
from hadronica.particles import species
from hadronica.processes import BEAMS, BeamMode

# Elementary species, then the mesons the decay library already knows.
COLLIDER_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Leptons", (11, -11, 13, -13, 15, -15, 12, -12, 14, -14, 16, -16)),
    ("Quarks", (2, -2, 1, -1, 3, -3, 4, -4, 5, -5, 6, -6)),
    ("Bosons", (22, 21, 23, 24, -24, 25)),
    ("Mesons", (111, 211, -211, 113, 213, -213, 20213, -20213)),
)


def energy_of(pdg: int, momentum: float) -> float:
    mass = species(pdg).mass
    p = max(0.0, momentum)
    return math.sqrt(p * p + mass * mass)


def beta_speed(pdg: int, momentum: float) -> float:
    energy = energy_of(pdg, momentum)
    if energy <= 0.0:
        return 0.0
    return max(0.0, momentum) / energy


def format_beta(beta: float) -> str:
    if beta >= 0.99995:
        return "β ≈ c"
    if beta >= 0.999:
        return f"β {beta:.5f} c"
    if beta <= 1.0e-4:
        return "at rest"
    return f"β {beta:.3f} c"


def equal_headon_momentum(mass: float, sqrt_s: float) -> float:
    """Momentum of each particle in an equal-momentum head-on collision."""
    half = 0.5 * sqrt_s
    if half <= mass:
        return 0.0
    return math.sqrt(half * half - mass * mass)


def incoming_momenta(
    pdg_a: int,
    momentum_a: float,
    pdg_b: int,
    momentum_b: float,
    angle_deg: float,
) -> tuple[FourVector, FourVector]:
    """Lab 4-vectors. A travels along +x. ``angle_deg`` is from A's momentum to B's.

    180° puts B along −x, which is a head-on collision in the plane of the display.
    """
    angle = math.radians(min(180.0, max(0.0, angle_deg)))
    p_a = max(0.0, momentum_a)
    p_b = max(0.0, momentum_b)
    a = FourVector(energy_of(pdg_a, p_a), p_a, 0.0, 0.0)
    b = FourVector(energy_of(pdg_b, p_b), p_b * math.cos(angle), p_b * math.sin(angle), 0.0)
    return a, b


def invariant_sqrt_s(p_a: FourVector, p_b: FourVector) -> float:
    return (p_a + p_b).mass


def beam_for_initial(pdg_a: int, pdg_b: int) -> tuple[BeamMode, bool] | None:
    """Return the supported beam and whether A is the generator's +z fermion."""
    for beam in BEAMS.values():
        if (pdg_a, pdg_b) == (beam.pdg_plus, beam.pdg_minus):
            return beam, True
        if (pdg_a, pdg_b) == (beam.pdg_minus, beam.pdg_plus):
            return beam, False
    return None


def _unit(x: float, y: float, z: float) -> tuple[float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1.0e-12:
        return (1.0, 0.0, 0.0)
    return (x / norm, y / norm, z / norm)


def rotate_z_onto(p: FourVector, direction: tuple[float, float, float]) -> FourVector:
    """Rotate ``p`` so the +z axis lands on ``direction``."""
    ux, uy, uz = direction
    if uz > 0.999999:
        return p
    if uz < -0.999999:
        return FourVector(p.e, p.px, -p.py, -p.pz)
    # Axis k = z × u, |k| = sin θ, cos θ = uz.
    sin_t = math.hypot(uy, ux)
    kx, ky, kz = -uy / sin_t, ux / sin_t, 0.0
    cos_t = uz
    cx = ky * p.pz - kz * p.py
    cy = kz * p.px - kx * p.pz
    cz = kx * p.py - ky * p.px
    k_dot = kx * p.px + ky * p.py + kz * p.pz
    one_minus = 1.0 - cos_t
    return FourVector(
        p.e,
        p.px * cos_t + cx * sin_t + kx * k_dot * one_minus,
        p.py * cos_t + cy * sin_t + ky * k_dot * one_minus,
        p.pz * cos_t + cz * sin_t + kz * k_dot * one_minus,
    )


def embed_in_lab(event: Event, p_a: FourVector, p_b: FourVector, a_is_plus: bool) -> Event:
    """Boost a center-of-mass event into the lab defined by the two incoming momenta.

    The generator emits the +z fermion along +z. That axis is rotated onto that
    fermion's momentum in the center of mass, then the whole event is boosted
    by the velocity of the center of mass.
    """
    plus_lab = p_a if a_is_plus else p_b
    minus_lab = p_b if a_is_plus else p_a
    total = plus_lab + minus_lab
    beta = total.beta()
    plus_cm = boost(plus_lab, (-beta[0], -beta[1], -beta[2]))
    direction = _unit(plus_cm.px, plus_cm.py, plus_cm.pz)
    rewritten: list[ParticleRecord] = []
    for index, particle in enumerate(event.particles):
        if index == 0:
            rewritten.append(ParticleRecord(particle.pdg, plus_lab, "beam", None))
            continue
        if index == 1:
            rewritten.append(ParticleRecord(particle.pdg, minus_lab, "beam", None))
            continue
        moved = boost(rotate_z_onto(particle.p4, direction), beta)
        rewritten.append(
            ParticleRecord(particle.pdg, moved, particle.status, particle.parent, particle.color)
        )
    event.particles = rewritten
    event.sqrt_s = total.mass
    _assert_conserved(event)
    return event
