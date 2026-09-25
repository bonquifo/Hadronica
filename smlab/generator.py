"""Build a complete, conservation-checked collision event."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from smlab.constants import ALPHA, GAMMA_Z, M_Z, PB_PER_GEV2
from smlab.decays import decay_once
from smlab.kinematics import (
    isr_exponent,
    isr_weight,
    recoil_against_photon,
    two_body_cm,
    two_body_momentum,
)
from smlab.lorentz import FourVector, boost_from_rest
from smlab.particles import is_quark, species
from smlab.processes import BEAMS, BeamMode, Process


@dataclass(slots=True)
class ParticleRecord:
    pdg: int
    p4: FourVector
    status: str  # beam, isr, intermediate, final
    parent: int | None
    color: int | None = None


@dataclass(slots=True)
class Event:
    seed: int
    sqrt_s: float
    sqrt_s_hat: float
    beam_id: str
    process_id: str
    process_title: str
    cos_theta: float
    phi: float
    isr_v: float
    particles: list[ParticleRecord] = field(default_factory=list)

    def finals(self) -> list[ParticleRecord]:
        return [p for p in self.particles if p.status == "final"]

    def beams(self) -> list[ParticleRecord]:
        return [p for p in self.particles if p.status == "beam"]

    def isr_photon(self) -> ParticleRecord | None:
        for particle in self.particles:
            if particle.status == "isr":
                return particle
        return None


@dataclass(frozen=True, slots=True)
class ConservationReport:
    delta_e: float
    delta_p: float
    delta_charge_thirds: int
    delta_baryon_thirds: int
    delta_lepton: tuple[int, int, int]
    missing_px: float
    missing_py: float

    @property
    def ok(self) -> bool:
        return (
            abs(self.delta_e) < 1.0e-6
            and self.delta_p < 1.0e-6
            and self.delta_charge_thirds == 0
            and self.delta_baryon_thirds == 0
            and self.delta_lepton == (0, 0, 0)
        )


class PhysicsError(RuntimeError):
    pass


def beam_four_vectors(sqrt_s: float, beam: BeamMode) -> tuple[FourVector, FourVector]:
    m_plus = species(beam.pdg_plus).mass
    m_minus = species(beam.pdg_minus).mass
    p = two_body_momentum(sqrt_s, m_plus, m_minus)
    e_plus = math.sqrt(p * p + m_plus * m_plus)
    e_minus = math.sqrt(p * p + m_minus * m_minus)
    return (
        FourVector(e_plus, 0.0, 0.0, p),
        FourVector(e_minus, 0.0, 0.0, -p),
    )


def conservation_report(event: Event) -> ConservationReport:
    beams = event.beams()
    if len(beams) != 2:
        raise PhysicsError("an event needs two beam particles")
    incoming = beams[0].p4 + beams[1].p4
    outgoing = FourVector(0.0, 0.0, 0.0, 0.0)
    charge = 0
    baryon = 0
    lepton = [0, 0, 0]
    visible_px = 0.0
    visible_py = 0.0
    for particle in event.finals():
        outgoing = outgoing + particle.p4
        spec = species(particle.pdg)
        charge += spec.charge_thirds
        baryon += spec.baryon_thirds
        for i in range(3):
            lepton[i] += spec.lepton[i]
        if abs(particle.pdg) not in (12, 14, 16):
            visible_px += particle.p4.px
            visible_py += particle.p4.py
    photon = event.isr_photon()
    if photon is not None:
        outgoing = outgoing + photon.p4
        visible_px += photon.p4.px
        visible_py += photon.p4.py
    for particle in beams:
        spec = species(particle.pdg)
        charge -= spec.charge_thirds
        baryon -= spec.baryon_thirds
        for i in range(3):
            lepton[i] -= spec.lepton[i]
    delta = outgoing - incoming
    return ConservationReport(
        delta_e=delta.e,
        delta_p=math.sqrt(delta.px**2 + delta.py**2 + delta.pz**2),
        delta_charge_thirds=charge,
        delta_baryon_thirds=baryon,
        delta_lepton=(lepton[0], lepton[1], lepton[2]),
        missing_px=incoming.px - visible_px,
        missing_py=incoming.py - visible_py,
    )


def _assert_conserved(event: Event) -> ConservationReport:
    report = conservation_report(event)
    if not report.ok:
        raise PhysicsError(
            f"conservation failed for {event.process_id} at √s={event.sqrt_s}: {report}"
        )
    return report


class IsrTable:
    """The ISR-folded integrand on a grid in u = v^β, where v is the photon energy fraction.

    The O(β) leading-log Kuraev–Fadin radiator is

        H(v) = β v^(β-1) (1 + 3β/4) - β (1 - v/2),

    which integrates to one. With u = v^β, H(v) dv = w(u) du and
    w(u) = (1 + 3β/4) - (1 - v/2) v^(1-β) is smooth and positive, so
    σ = ∫₀¹ w(u) σ̂(s(1-v)) du. Extra nodes are placed across the Z line shape,
    because radiative return puts a narrow peak at s' = M_Z².
    """

    def __init__(self, born, sqrt_s: float, beam_mass: float, n: int = 480):
        beta = isr_exponent(sqrt_s, beam_mass, ALPHA)
        self.beta = beta
        self.sqrt_s = sqrt_s
        s = sqrt_s * sqrt_s
        us = set(np.linspace(0.0, 1.0, max(16, n)).tolist())
        # Hard photons: σ̂ grows like 1/s' toward the pair threshold, so also
        # place nodes evenly in ln s' down to 10⁻⁴ s.
        for s_prime in np.geomspace(1.0e-4 * s, s, max(16, n // 2), endpoint=False):
            us.add((1.0 - float(s_prime) / s) ** beta)
        if s > M_Z * M_Z:
            # s' = M_Z² + M_Z Γ_Z tan φ, φ uniform: dense where the Breit–Wigner is.
            g = M_Z * GAMMA_Z
            lo = math.atan((0.0 - M_Z * M_Z) / g)
            hi = math.atan((s - M_Z * M_Z) / g)
            for phi in np.linspace(lo, hi, 241)[1:-1]:
                s_prime = M_Z * M_Z + g * math.tan(float(phi))
                v = 1.0 - s_prime / s
                if 0.0 < v < 1.0:
                    us.add(v**beta)
        grid = np.array(sorted(us), dtype=float)
        values = np.empty_like(grid)
        shats = np.empty_like(grid)
        for i, u in enumerate(grid):
            v = u ** (1.0 / beta) if u > 0.0 else 0.0
            if v >= 1.0:
                shats[i] = 0.0
                values[i] = 0.0
                continue
            shat = sqrt_s * math.sqrt(1.0 - v)
            shats[i] = shat
            values[i] = isr_weight(v, beta) * born(shat)
        self.u = grid
        self.values = values
        areas = 0.5 * (values[1:] + values[:-1]) * np.diff(grid)
        self.cdf = np.concatenate(([0.0], np.cumsum(areas)))
        self.total = float(self.cdf[-1])

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        """Return (√s', v) from the folded distribution, continuous inside each cell."""
        if self.total <= 0.0:
            raise PhysicsError("process is closed at every radiator scale")
        target = float(rng.random()) * self.total
        k = int(np.searchsorted(self.cdf, target, side="right")) - 1
        k = min(max(k, 0), self.u.size - 2)
        need = target - self.cdf[k]
        h = self.u[k + 1] - self.u[k]
        y0, y1 = self.values[k], self.values[k + 1]
        slope = (y1 - y0) / h if h > 0.0 else 0.0
        if abs(slope) * h < 1.0e-12 * max(y0, y1, 1.0e-300):
            du = need / y0 if y0 > 0.0 else 0.5 * h
        else:
            disc = max(y0 * y0 + 2.0 * slope * need, 0.0)
            du = (math.sqrt(disc) - y0) / slope
        u = min(max(self.u[k] + du, self.u[k]), self.u[k + 1])
        v = u ** (1.0 / self.beta) if u > 0.0 else 0.0
        v = min(v, 1.0 - 1.0e-12)
        return self.sqrt_s * math.sqrt(1.0 - v), v


_ISR_TABLES: dict[tuple, IsrTable] = {}


def isr_table(process: Process, beam: BeamMode, sqrt_s: float) -> IsrTable:
    """Cached radiator grid for this process, beam, and energy."""
    key = (process.id, beam.id, round(sqrt_s, 9))
    table = _ISR_TABLES.get(key)
    if table is None:
        if len(_ISR_TABLES) > 256:
            _ISR_TABLES.clear()
        table = IsrTable(lambda shat: process.born_sigma(beam, shat), sqrt_s, species(beam.pdg_plus).mass)
        _ISR_TABLES[key] = table
    return table


def convolute_born(
    born,
    sqrt_s: float,
    beam_mass: float,
    n: int = 480,
) -> float:
    """O(β) leading-log ISR convolution of a Born cross section, in GeV^-2."""
    return IsrTable(born, sqrt_s, beam_mass, n).total


def _expand(
    pdg: int,
    p4: FourVector,
    parent: int | None,
    records: list[ParticleRecord],
    rng: np.random.Generator,
    force_muon: bool,
    color: int | None,
) -> None:
    daughters = decay_once(pdg, p4, rng, force_muon=force_muon)
    status = "intermediate" if daughters else "final"
    records.append(ParticleRecord(pdg, p4, status, parent, color))
    if not daughters:
        return
    index = len(records) - 1
    child_colors = _daughter_colors(daughters, rng)
    for (child_pdg, child_p4), child_color in zip(daughters, child_colors):
        _expand(child_pdg, child_p4, index, records, rng, force_muon, child_color)


def _daughter_colors(daughters, rng: np.random.Generator) -> list[int | None]:
    if not any(is_quark(pdg) for pdg, _p4 in daughters):
        return [None] * len(daughters)
    color = int(rng.integers(1, 4))
    out: list[int | None] = []
    for pdg, _p4 in daughters:
        if is_quark(pdg):
            out.append(color if pdg > 0 else -color)
        else:
            out.append(None)
    return out


def generate_event(
    process: Process,
    beam: BeamMode | str,
    sqrt_s: float,
    seed: int,
    *,
    isr: bool = True,
    force_muon: bool = False,
) -> Event:
    """Generate one unweighted event. Raises PhysicsError if a law is violated."""
    if isinstance(beam, str):
        beam = BEAMS[beam]
    if sqrt_s <= 0.0:
        raise PhysicsError("√s must be positive")
    if not process.allowed(beam, sqrt_s):
        raise PhysicsError(f"{process.id} is closed for {beam.id} at √s = {sqrt_s}")

    rng = np.random.default_rng(seed)
    use_isr = bool(isr) and beam.id in ("ee", "mumu")
    if use_isr:
        table = isr_table(process, beam, sqrt_s)
        for _attempt in range(64):
            shat, v = table.sample(rng)
            if process.allowed(beam, shat):
                break
        plus_z = bool(rng.random() < 0.5)
    else:
        shat, v = sqrt_s, 0.0
        plus_z = True

    if not process.allowed(beam, shat):
        raise PhysicsError("sampled below the hard threshold")
    scatter = process.sample(beam, shat, rng)
    m1 = _hard_mass(scatter.pdg1, shat)
    m2 = _hard_mass(scatter.pdg2, shat)
    p1_cm, p2_cm = two_body_cm(shat, m1, m2, scatter.cos_theta, scatter.phi)

    records: list[ParticleRecord] = []
    b_plus, b_minus = beam_four_vectors(sqrt_s, beam)
    records.append(ParticleRecord(beam.pdg_plus, b_plus, "beam", None))
    records.append(ParticleRecord(beam.pdg_minus, b_minus, "beam", None))

    if v > 0.0:
        photon, recoil = recoil_against_photon(sqrt_s, v, plus_z)
        records.append(ParticleRecord(22, photon, "isr", None))
        p1 = boost_from_rest(p1_cm, recoil)
        p2 = boost_from_rest(p2_cm, recoil)
    else:
        p1, p2 = p1_cm, p2_cm

    colors = _daughter_colors([(scatter.pdg1, p1), (scatter.pdg2, p2)], rng)
    _expand(scatter.pdg1, p1, None, records, rng, force_muon, colors[0])
    _expand(scatter.pdg2, p2, None, records, rng, force_muon, colors[1])

    event = Event(
        seed=seed,
        sqrt_s=sqrt_s,
        sqrt_s_hat=shat,
        beam_id=beam.id,
        process_id=process.id,
        process_title=f"{beam.label} {process.title}",
        cos_theta=scatter.cos_theta,
        phi=scatter.phi,
        isr_v=v,
        particles=records,
    )
    _assert_conserved(event)
    return event


def _hard_mass(pdg: int, shat: float) -> float:
    """On-shell mass of a primary product. Photons and gluons stay massless."""
    del shat
    return species(pdg).mass


def sigma_pb(process: Process, beam: BeamMode | str, sqrt_s: float, isr: bool) -> float:
    """Cross section in picobarns, optionally convolved with O(β) leading-log ISR."""
    if isinstance(beam, str):
        beam = BEAMS[beam]
    if not process.allowed(beam, sqrt_s):
        return 0.0
    if isr and beam.id in ("ee", "mumu"):
        value = isr_table(process, beam, sqrt_s).total
    else:
        value = process.born_sigma(beam, sqrt_s)
    return value * PB_PER_GEV2
