"""Minkowski 4-vectors with metric (+, -, -, -). Energies are in GeV."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FourVector:
    e: float
    px: float
    py: float
    pz: float

    def __add__(self, other: FourVector) -> FourVector:
        return FourVector(
            self.e + other.e,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
        )

    def __sub__(self, other: FourVector) -> FourVector:
        return FourVector(
            self.e - other.e,
            self.px - other.px,
            self.py - other.py,
            self.pz - other.pz,
        )

    def __mul__(self, scale: float) -> FourVector:
        return FourVector(self.e * scale, self.px * scale, self.py * scale, self.pz * scale)

    __rmul__ = __mul__

    def dot(self, other: FourVector) -> float:
        return (
            self.e * other.e
            - self.px * other.px
            - self.py * other.py
            - self.pz * other.pz
        )

    @property
    def m2(self) -> float:
        return self.dot(self)

    @property
    def mass(self) -> float:
        m2 = self.m2
        if m2 < 0.0:
            if m2 > -1.0e-8:
                return 0.0
            raise ValueError(f"spacelike 4-vector, m^2 = {m2}")
        return math.sqrt(m2)

    @property
    def p(self) -> float:
        return math.sqrt(self.px * self.px + self.py * self.py + self.pz * self.pz)

    @property
    def pt(self) -> float:
        return math.hypot(self.px, self.py)

    @property
    def phi(self) -> float:
        return math.atan2(self.py, self.px)

    @property
    def pz_signed(self) -> float:
        return self.pz

    def beta(self) -> tuple[float, float, float]:
        if self.e <= 0.0:
            raise ValueError("cannot boost with a non-positive energy")
        return (self.px / self.e, self.py / self.e, self.pz / self.e)

    def almost_equal(self, other: FourVector, tol: float) -> bool:
        return (
            abs(self.e - other.e) <= tol
            and abs(self.px - other.px) <= tol
            and abs(self.py - other.py) <= tol
            and abs(self.pz - other.pz) <= tol
        )


def boost(p: FourVector, beta: tuple[float, float, float]) -> FourVector:
    """Boost ``p`` by velocity ``beta`` (the new frame sees the old origin moving at -beta).

    A particle at rest, boosted by the parent's lab velocity, becomes the parent.
    """
    bx, by, bz = beta
    b2 = bx * bx + by * by + bz * bz
    if b2 < 1.0e-18:
        return p
    if b2 >= 1.0:
        raise ValueError(f"boost velocity is not timelike: beta^2 = {b2}")
    gamma = 1.0 / math.sqrt(1.0 - b2)
    bp = bx * p.px + by * p.py + bz * p.pz
    # p' = p + [(gamma - 1) (beta·p) / beta^2 + gamma E] beta
    factor = (gamma - 1.0) * bp / b2 + gamma * p.e
    return FourVector(
        gamma * (p.e + bp),
        p.px + factor * bx,
        p.py + factor * by,
        p.pz + factor * bz,
    )


def boost_from_rest(p_rest: FourVector, parent: FourVector) -> FourVector:
    """Take a 4-vector in ``parent``'s rest frame into the frame where ``parent`` is given."""
    return boost(p_rest, parent.beta())
