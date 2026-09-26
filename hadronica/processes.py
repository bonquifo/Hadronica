"""Scattering processes that can be selected in the laboratory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hadronica.constants import M_H, M_Z, QUARK_CONTINUUM_GEV, THETA_FIDUCIAL
from hadronica.electroweak import (
    bhabha_dsigma_domega,
    bhabha_sigma,
    diphoton_dsigma_domega,
    diphoton_sigma,
    fermion_angular_density,
    fermion_pair_amplitudes,
    fermion_pair_sigma,
    higgsstrahlung_density,
    higgsstrahlung_sigma,
    neutrino_w_dsigma_domega,
    neutrino_w_sigma,
)
from hadronica.particles import is_quark, species


@dataclass(frozen=True, slots=True)
class BeamMode:
    id: str
    pdg_plus: int  # travels along +z; θ is measured from this direction
    pdg_minus: int
    label: str


BEAMS: dict[str, BeamMode] = {
    "ee": BeamMode("ee", 11, -11, "e⁻ × e⁺"),
    "mumu": BeamMode("mumu", 13, -13, "μ⁻ × μ⁺"),
    "uu": BeamMode("uu", 2, -2, "u × ū"),
    "dd": BeamMode("dd", 1, -1, "d × d̄"),
    "ss": BeamMode("ss", 3, -3, "s × s̄"),
    "cc": BeamMode("cc", 4, -4, "c × c̄"),
    "bb": BeamMode("bb", 5, -5, "b × b̄"),
}


def pair_threshold(pdg: int) -> float:
    mass = species(abs(pdg)).mass
    floor = QUARK_CONTINUUM_GEV.get(abs(pdg), 0.0) if is_quark(pdg) else 0.0
    return max(2.0 * mass, floor)


@dataclass(frozen=True, slots=True)
class HardScatter:
    pdg1: int
    pdg2: int
    cos_theta: float
    phi: float


class Process:
    id: str
    title: str
    family: str  # "signal" or "qed"
    blurb: str

    def allowed(self, beam: BeamMode, sqrt_s: float) -> bool:
        raise NotImplementedError

    def born_sigma(self, beam: BeamMode, sqrt_s: float) -> float:
        """Born cross section in GeV^-2 at the hard scale."""
        raise NotImplementedError

    def sample(self, beam: BeamMode, sqrt_s: float, rng: np.random.Generator) -> HardScatter:
        raise NotImplementedError


def _cdf_sample(fn, c_min: float, c_max: float, rng: np.random.Generator, n: int = 2400) -> float:
    """Sample cosθ from an un-normalized density by inverting its trapezoid CDF.

    Inside each grid cell the density is taken as linear, so the inverse is
    continuous rather than snapped to grid points.
    """
    cs = np.linspace(c_min, c_max, n)
    ys = np.empty(n, dtype=float)
    for i, c in enumerate(cs):
        ys[i] = max(float(fn(float(c))), 0.0)
    h = cs[1] - cs[0]
    areas = 0.5 * (ys[1:] + ys[:-1]) * h
    total = float(areas.sum())
    if total <= 0.0:
        raise RuntimeError("angular density vanished")
    cdf = np.concatenate(([0.0], np.cumsum(areas)))
    target = float(rng.random()) * total
    k = int(np.searchsorted(cdf, target, side="right")) - 1
    k = min(max(k, 0), n - 2)
    need = target - cdf[k]
    y0, y1 = ys[k], ys[k + 1]
    slope = (y1 - y0) / h
    if abs(slope) < 1.0e-14 * max(y0, y1, 1.0e-300):
        dx = need / y0 if y0 > 0.0 else 0.5 * h
    else:
        # Solve y0 dx + slope dx²/2 = need for the root inside the cell.
        disc = max(y0 * y0 + 2.0 * slope * need, 0.0)
        dx = (math.sqrt(disc) - y0) / slope
    return float(min(max(cs[k] + dx, c_min), c_max))


class FermionPair(Process):
    def __init__(self, final_pdg: int):
        self.final_pdg = final_pdg
        self.id = f"ff{final_pdg}"
        name = species(final_pdg).name
        anti = species(-final_pdg).name
        self.title = f"→ {name} {anti}"
        self.family = "signal"
        if is_quark(final_pdg):
            self.blurb = "Improved Born γ*/Z* → quark pair, parton level. No hadronization."
        elif abs(final_pdg) in (12, 14, 16):
            self.blurb = (
                "Z* → neutrino pair (plus t-channel W when the beam is the same lepton flavor). "
                "Visible only as missing momentum."
            )
        else:
            self.blurb = "Improved Born γ*/Z* → lepton pair, with vector and axial mass thresholds."

    def allowed(self, beam: BeamMode, sqrt_s: float) -> bool:
        # Same-flavor charged leptons are Bhabha / Møller, not this s-channel formula.
        if abs(beam.pdg_plus) == abs(self.final_pdg) and abs(self.final_pdg) in (11, 13):
            return False
        if is_quark(beam.pdg_plus) and is_quark(self.final_pdg):
            return False
        if is_quark(beam.pdg_plus) and sqrt_s < 8.0:
            return False
        return sqrt_s > pair_threshold(self.final_pdg) + 1.0e-6

    def has_w_exchange(self, beam: BeamMode) -> bool:
        """ℓ⁻ℓ⁺ → ν_ℓ ν̄_ℓ also proceeds through t-channel W exchange."""
        lepton = abs(beam.pdg_plus)
        return lepton in (11, 13) and abs(self.final_pdg) == lepton + 1

    def born_sigma(self, beam: BeamMode, sqrt_s: float) -> float:
        if not self.allowed(beam, sqrt_s):
            return 0.0
        if self.has_w_exchange(beam):
            return neutrino_w_sigma(sqrt_s)
        return fermion_pair_sigma(sqrt_s, beam.pdg_plus, self.final_pdg)

    def sample(self, beam: BeamMode, sqrt_s: float, rng: np.random.Generator) -> HardScatter:
        if self.has_w_exchange(beam):
            cos_theta = _cdf_sample(lambda c: neutrino_w_dsigma_domega(sqrt_s, c), -1.0, 1.0, rng, n=1200)
            phi = float(rng.uniform(0.0, 2.0 * math.pi))
            return HardScatter(self.final_pdg, -self.final_pdg, cos_theta, phi)
        a_v, a_a, a_1, beta = fermion_pair_amplitudes(sqrt_s, beam.pdg_plus, self.final_pdg)
        if beta <= 0.0:
            raise RuntimeError("fermion pair is below threshold")

        def density(c: float) -> float:
            return fermion_angular_density(c, a_v, a_a, a_1, beta)

        cos_theta = _cdf_sample(density, -1.0, 1.0, rng, n=1200)
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        return HardScatter(self.final_pdg, -self.final_pdg, cos_theta, phi)


class Bhabha(Process):
    def __init__(self) -> None:
        self.id = "bhabha"
        self.title = "→ e⁻ e⁺   Bhabha"
        self.family = "qed"
        self.blurb = "QED Bhabha scattering (photon exchange only) with a 10° fiducial cut. Z exchange is omitted."

    def allowed(self, beam: BeamMode, sqrt_s: float) -> bool:
        return beam.id == "ee" and sqrt_s > 1.0

    def born_sigma(self, beam: BeamMode, sqrt_s: float) -> float:
        if not self.allowed(beam, sqrt_s):
            return 0.0
        return bhabha_sigma(sqrt_s)

    def sample(self, beam: BeamMode, sqrt_s: float, rng: np.random.Generator) -> HardScatter:
        c_max = math.cos(THETA_FIDUCIAL)

        def density(c: float) -> float:
            return bhabha_dsigma_domega(sqrt_s, c)

        cos_theta = _cdf_sample(density, -c_max, c_max, rng, n=6000)
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        return HardScatter(11, -11, cos_theta, phi)


class Diphoton(Process):
    def __init__(self) -> None:
        self.id = "diphoton"
        self.title = "→ γ γ"
        self.family = "qed"
        self.blurb = "QED annihilation to two photons, massless Born result, 10° fiducial cut."

    def allowed(self, beam: BeamMode, sqrt_s: float) -> bool:
        return beam.id == "ee" and sqrt_s > 1.0

    def born_sigma(self, beam: BeamMode, sqrt_s: float) -> float:
        if not self.allowed(beam, sqrt_s):
            return 0.0
        return diphoton_sigma(sqrt_s)

    def sample(self, beam: BeamMode, sqrt_s: float, rng: np.random.Generator) -> HardScatter:
        c_max = math.cos(THETA_FIDUCIAL)

        def density(c: float) -> float:
            return diphoton_dsigma_domega(sqrt_s, c)

        cos_theta = _cdf_sample(density, -c_max, c_max, rng, n=4000)
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        return HardScatter(22, 22, cos_theta, phi)


class Higgsstrahlung(Process):
    def __init__(self) -> None:
        self.id = "zh"
        self.title = "→ Z H"
        self.family = "signal"
        self.blurb = "Born Higgsstrahlung. Z and H decay; H→WW* and H→ZZ* are off-shell."

    def allowed(self, beam: BeamMode, sqrt_s: float) -> bool:
        return beam.id in ("ee", "mumu") and sqrt_s > M_Z + M_H

    def born_sigma(self, beam: BeamMode, sqrt_s: float) -> float:
        if not self.allowed(beam, sqrt_s):
            return 0.0
        # The compact formula is written for electrons. A muon collider has the
        # same Born structure up to the negligible muon-mass correction.
        return higgsstrahlung_sigma(sqrt_s)

    def sample(self, beam: BeamMode, sqrt_s: float, rng: np.random.Generator) -> HardScatter:
        def density(c: float) -> float:
            return higgsstrahlung_density(c, sqrt_s)

        cos_theta = _cdf_sample(density, -1.0, 1.0, rng, n=800)
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        return HardScatter(23, 25, cos_theta, phi)


def all_processes() -> list[Process]:
    finals = (13, 15, 12, 14, 16, 2, 1, 3, 4, 5, 6, 11)
    processes: list[Process] = [FermionPair(pdg) for pdg in finals]
    processes.append(Bhabha())
    processes.append(Diphoton())
    processes.append(Higgsstrahlung())
    return processes


def process_by_id(process_id: str) -> Process:
    for process in all_processes():
        if process.id == process_id:
            return process
    raise KeyError(process_id)
