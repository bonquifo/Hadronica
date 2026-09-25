"""Standard Model species used by the generator, keyed by PDG id."""

from __future__ import annotations

from dataclasses import dataclass

from smlab.constants import (
    C_TAU_MU_M,
    C_TAU_PI_M,
    C_TAU_TAU_M,
    M_A1,
    M_B,
    M_C,
    M_D,
    M_E,
    M_H,
    M_MU,
    M_PI,
    M_PI0,
    M_RHO,
    M_RHO0,
    M_S,
    M_T,
    M_TAU,
    M_U,
    M_W,
    M_Z,
)


@dataclass(frozen=True, slots=True)
class Species:
    pdg: int
    name: str
    mass: float
    charge: float
    spin: str
    baryon_thirds: int
    # Lepton numbers (L_e, L_mu, L_tau).
    lepton: tuple[int, int, int]
    kind: str
    ctau_m: float | None = None

    @property
    def charge_thirds(self) -> int:
        return int(round(self.charge * 3.0))


def _anti(species: Species) -> Species:
    e, mu, tau = species.lepton
    bar = species.name
    if species.name.endswith("⁻"):
        bar = species.name[:-1] + "⁺"
    elif species.name.endswith("⁺"):
        bar = species.name[:-1] + "⁻"
    elif species.name.startswith("ν"):
        bar = "ν̄" + species.name[1:]
    else:
        bar = species.name + "̄"
    return Species(
        pdg=-species.pdg,
        name=bar,
        mass=species.mass,
        charge=-species.charge,
        spin=species.spin,
        baryon_thirds=-species.baryon_thirds,
        lepton=(-e, -mu, -tau),
        kind=species.kind,
        ctau_m=species.ctau_m,
    )


_PARTICLES: tuple[Species, ...] = (
    Species(11, "e⁻", M_E, -1.0, "1/2", 0, (1, 0, 0), "lepton"),
    Species(12, "νe", 0.0, 0.0, "1/2", 0, (1, 0, 0), "lepton"),
    Species(13, "μ⁻", M_MU, -1.0, "1/2", 0, (0, 1, 0), "lepton", C_TAU_MU_M),
    Species(14, "νμ", 0.0, 0.0, "1/2", 0, (0, 1, 0), "lepton"),
    Species(15, "τ⁻", M_TAU, -1.0, "1/2", 0, (0, 0, 1), "lepton", C_TAU_TAU_M),
    Species(16, "ντ", 0.0, 0.0, "1/2", 0, (0, 0, 1), "lepton"),
    Species(1, "d", M_D, -1.0 / 3.0, "1/2", 1, (0, 0, 0), "quark"),
    Species(2, "u", M_U, 2.0 / 3.0, "1/2", 1, (0, 0, 0), "quark"),
    Species(3, "s", M_S, -1.0 / 3.0, "1/2", 1, (0, 0, 0), "quark"),
    Species(4, "c", M_C, 2.0 / 3.0, "1/2", 1, (0, 0, 0), "quark"),
    Species(5, "b", M_B, -1.0 / 3.0, "1/2", 1, (0, 0, 0), "quark"),
    Species(6, "t", M_T, 2.0 / 3.0, "1/2", 1, (0, 0, 0), "quark"),
    Species(21, "g", 0.0, 0.0, "1", 0, (0, 0, 0), "boson"),
    Species(22, "γ", 0.0, 0.0, "1", 0, (0, 0, 0), "boson"),
    Species(23, "Z", M_Z, 0.0, "1", 0, (0, 0, 0), "boson"),
    Species(24, "W⁺", M_W, 1.0, "1", 0, (0, 0, 0), "boson"),
    Species(25, "H", M_H, 0.0, "0", 0, (0, 0, 0), "boson"),
    Species(111, "π⁰", M_PI0, 0.0, "0", 0, (0, 0, 0), "meson"),
    Species(211, "π⁺", M_PI, 1.0, "0", 0, (0, 0, 0), "meson", C_TAU_PI_M),
    Species(113, "ρ⁰", M_RHO0, 0.0, "1", 0, (0, 0, 0), "meson"),
    Species(213, "ρ⁺", M_RHO, 1.0, "1", 0, (0, 0, 0), "meson"),
    Species(20213, "a₁⁺", M_A1, 1.0, "1", 0, (0, 0, 0), "meson"),
)

CATALOG: dict[int, Species] = {}
for _species in _PARTICLES:
    CATALOG[_species.pdg] = _species
    if _species.pdg not in (21, 22, 23, 25, 111, 113):
        CATALOG[-_species.pdg] = _anti(_species)


def species(pdg: int) -> Species:
    try:
        return CATALOG[pdg]
    except KeyError as exc:
        raise KeyError(f"unknown PDG id {pdg}") from exc


def is_quark(pdg: int) -> bool:
    return abs(pdg) in (1, 2, 3, 4, 5, 6)


def is_neutrino(pdg: int) -> bool:
    return abs(pdg) in (12, 14, 16)


def is_charged_lepton(pdg: int) -> bool:
    return abs(pdg) in (11, 13, 15)


def weak_isospin(pdg: int) -> float:
    """Third component of weak isospin for a fermion (the particle, not the anti)."""
    a = abs(pdg)
    if a in (2, 4, 6, 12, 14, 16):
        return 0.5
    if a in (1, 3, 5, 11, 13, 15):
        return -0.5
    raise ValueError(f"PDG {pdg} is not an SM fermion")


def color_count(pdg: int) -> int:
    return 3 if is_quark(pdg) else 1
