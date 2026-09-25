"""Two-body phase space and collinear initial-state radiation kinematics."""

from __future__ import annotations

import math

from smlab.lorentz import FourVector, boost_from_rest


def kallen(x: float, y: float, z: float) -> float:
    """Källén function λ(x, y, z)."""
    return x * x + y * y + z * z - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def two_body_momentum(sqrt_s: float, m1: float, m2: float) -> float:
    """Magnitude of each 3-momentum in the center-of-momentum frame."""
    if sqrt_s < 0.0:
        raise ValueError("negative center-of-mass energy")
    s = sqrt_s * sqrt_s
    if sqrt_s < m1 + m2:
        lam = kallen(s, m1 * m1, m2 * m2)
        if lam > -1.0e-6 * max(s * s, 1.0):
            return 0.0
        raise ValueError("below the two-body threshold")
    lam = kallen(s, m1 * m1, m2 * m2)
    if lam < 0.0:
        lam = 0.0
    return math.sqrt(lam) / (2.0 * sqrt_s)


def two_body_cm(
    sqrt_s: float,
    m1: float,
    m2: float,
    cos_theta: float,
    phi: float,
) -> tuple[FourVector, FourVector]:
    """Back-to-back 4-momenta. Particle 1 is emitted at polar angle ``cos_theta``."""
    if cos_theta < -1.0 - 1.0e-9 or cos_theta > 1.0 + 1.0e-9:
        raise ValueError(f"cos theta out of range: {cos_theta}")
    c = min(1.0, max(-1.0, cos_theta))
    p = two_body_momentum(sqrt_s, m1, m2)
    sin_theta = math.sqrt(max(0.0, 1.0 - c * c))
    px = p * sin_theta * math.cos(phi)
    py = p * sin_theta * math.sin(phi)
    pz = p * c
    e1 = math.sqrt(p * p + m1 * m1)
    e2 = math.sqrt(p * p + m2 * m2)
    return FourVector(e1, px, py, pz), FourVector(e2, -px, -py, -pz)


def two_body_decay(
    parent: FourVector,
    m1: float,
    m2: float,
    cos_theta: float,
    phi: float,
) -> tuple[FourVector, FourVector]:
    """Decay ``parent`` to two particles. Angles are in the parent rest frame."""
    m = parent.mass
    p1, p2 = two_body_cm(m, m1, m2, cos_theta, phi)
    return boost_from_rest(p1, parent), boost_from_rest(p2, parent)


def velocity_parameter(sqrt_s: float, mass: float) -> float:
    """β = sqrt(1 - 4 m^2 / s) for an equal-mass pair. Zero below threshold."""
    if sqrt_s <= 2.0 * mass:
        return 0.0
    return math.sqrt(1.0 - 4.0 * mass * mass / (sqrt_s * sqrt_s))


def isr_exponent(sqrt_s: float, beam_mass: float, alpha: float) -> float:
    """Leading-log radiator exponent β_e = (2α/π) (ln(s/m^2) - 1)."""
    if sqrt_s <= 0.0 or beam_mass <= 0.0:
        raise ValueError("ISR requires a positive energy and beam mass")
    log_term = math.log((sqrt_s * sqrt_s) / (beam_mass * beam_mass)) - 1.0
    return max((2.0 * alpha / math.pi) * log_term, 1.0e-8)


def isr_weight(v: float, beta: float) -> float:
    """Radiator weight per unit u = v^β for the O(β) Kuraev–Fadin structure function.

    H(v) = β v^(β-1) (1 + 3β/4) - β (1 - v/2)   [E. A. Kuraev, V. S. Fadin,
    Sov. J. Nucl. Phys. 41 (1985) 466], and β v^(β-1) dv = du, so
    H(v) dv = [(1 + 3β/4) - (1 - v/2) v^(1-β)] du. The weight is positive on
    0 ≤ v ≤ 1 and ∫₀¹ H(v) dv = 1.
    """
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"ISR fraction out of range: {v}")
    return (1.0 + 0.75 * beta) - (1.0 - 0.5 * v) * v ** (1.0 - beta)


def recoil_against_photon(sqrt_s: float, v: float, plus_z: bool) -> tuple[FourVector, FourVector]:
    """Collinear photon and the recoiling hard system.

    The photon carries energy fraction v of the beam energy (E_γ = v √s / 2) and
    travels along the beam. The hard system has invariant mass √(s(1-v)).
    """
    if not 0.0 <= v < 1.0:
        raise ValueError(f"ISR fraction out of range: {v}")
    e_gamma = 0.5 * v * sqrt_s
    sign = 1.0 if plus_z else -1.0
    photon = FourVector(e_gamma, 0.0, 0.0, sign * e_gamma)
    # Total incoming 4-momentum is (√s, 0, 0, 0) for symmetric beams.
    recoil = FourVector(sqrt_s - e_gamma, 0.0, 0.0, -sign * e_gamma)
    return photon, recoil
