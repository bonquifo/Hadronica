"""Decays of SM particles produced by the generator.

Z and W branching fractions are computed from the same couplings as the
production amplitudes (CKM magnitudes for the W, with a final-state QCD factor
on quark channels). Higgs branching fractions are the Standard Model table at
m_H = 125 GeV compiled by the LHC Higgs Cross Section Working Group and quoted
by the PDG review; they are renormalized so the listed modes sum to one.

Tau decays use PDG exclusive fractions for the leptonic, one-pion, rho, and a1
channels. The remaining few percent (mostly kaons and higher multiplicity) is
a charge-conserving four-pion state so 4-momentum and charge stay exact.

Leptonic muon and tau decays use the massless Michel spectrum. Decay angles of
W, Z, and H are isotropic in the parent rest frame: V-A angular correlations
are not simulated. H → V V* samples virtual masses from Breit–Wigner
propagators in q² times two-body phase space.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from smlab.constants import (
    CKM_ABS,
    GAMMA_A1,
    GAMMA_RHO,
    GAMMA_W,
    GAMMA_Z,
    M_A1,
    M_RHO,
    M_W,
    M_Z,
)
from smlab.electroweak import alpha_s, z_partial_widths
from smlab.kinematics import kallen, two_body_decay
from smlab.lorentz import FourVector, boost_from_rest
from smlab.particles import is_quark, species

# SM Higgs branching fractions at m_H = 125 GeV (PDG review / LHC HXSWG).
# gg is the loop-induced mode from that compilation. The table is renormalized
# to unity; the shift is far below the quoted theory uncertainties.
_HIGGS_BR_RAW: tuple[tuple[str, float], ...] = (
    ("bb", 0.582),
    ("ww", 0.214),
    ("gg", 0.0818),
    ("tautau", 0.0627),
    ("cc", 0.0289),
    ("zz", 0.0262),
    ("gammagamma", 0.00227),
    ("zgamma", 0.00153),
    ("mumu", 2.18e-4),
)
_HIGGS_SUM = sum(br for _name, br in _HIGGS_BR_RAW)
HIGGS_BR: tuple[tuple[str, float], ...] = tuple(
    (name, br / _HIGGS_SUM) for name, br in _HIGGS_BR_RAW
)

# PDG exclusive tau branching fractions (particle, not percent). The last entry
# absorbs kaon and high-multiplicity modes.
TAU_BR: tuple[tuple[str, float], ...] = (
    ("enu", 0.1782),
    ("munu", 0.1739),
    ("pi", 0.1082),
    ("rho", 0.2549),
    ("a1_neutral", 0.0926),  # τ → π 2π0 ν via a1 → ρ π0
    ("a1_charged", 0.0899),  # τ → 3π ν via a1 → ρ0 π
    ("four_pi", 0.0462),
    ("rest", 0.0561),
)


def _pick(weights: Sequence[float], rng: np.random.Generator) -> int:
    total = float(sum(weights))
    if total <= 0.0:
        raise RuntimeError("no open decay channel")
    u = float(rng.random()) * total
    acc = 0.0
    for i, weight in enumerate(weights):
        acc += weight
        if u <= acc:
            return i
    return len(weights) - 1


def _angles(rng: np.random.Generator) -> tuple[float, float]:
    cos_theta = float(rng.uniform(-1.0, 1.0))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    return cos_theta, phi


def w_channel_weights(mass: float) -> list[tuple[float, int, int]]:
    """Open W+ channels as (weight, pdg_up_or_lepton+, pdg_down_or_neutrino).

    The returned PDG ids are the W+ daughters. Charge conjugation is applied
    by the caller for W-.
    """
    qcd = 1.0 + alpha_s(max(mass, M_W)) / math.pi
    # Leptonic channels are pure V-A with weight 1 in the massless limit.
    channels: list[tuple[int, int, float]] = [
        (-11, 12, 1.0),
        (-13, 14, 1.0),
        (-15, 16, 1.0),
    ]
    out: list[tuple[float, int, int]] = []
    for pdg_a, pdg_b, weight in channels:
        if mass > species(pdg_a).mass + species(pdg_b).mass:
            out.append((weight, pdg_a, pdg_b))
    for (up, down), v_abs in CKM_ABS.items():
        m1 = species(up).mass
        m2 = species(down).mass
        if mass <= m1 + m2:
            continue
        # Momentum fraction relative to the massless limit, so heavy channels
        # close smoothly. The massless V-A color weight is 3 |V|^2 (1+α_s/π).
        p = _momentum_fraction(mass, m1, m2)
        out.append((3.0 * qcd * v_abs * v_abs * p, up, -down))
    return out


def _momentum_fraction(mass: float, m1: float, m2: float) -> float:
    lam = kallen(mass * mass, m1 * m1, m2 * m2)
    if lam <= 0.0:
        return 0.0
    p = math.sqrt(lam) / (2.0 * mass)
    return (2.0 * p) / mass


def z_channel_weights(mass: float) -> list[tuple[float, int, int]]:
    """Open Z channels as (width, fermion, antifermion)."""
    out = []
    for pdg, width in z_partial_widths(mass).items():
        if width > 0.0 and mass > 2.0 * species(pdg).mass:
            out.append((width, pdg, -pdg))
    return out


def branching_table_w(mass: float = M_W) -> list[tuple[float, int, int]]:
    raw = w_channel_weights(mass)
    total = sum(w for w, _a, _b in raw)
    return [(w / total, a, b) for w, a, b in raw]


def branching_table_z(mass: float = M_Z) -> list[tuple[float, int, int]]:
    raw = z_channel_weights(mass)
    total = sum(w for w, _a, _b in raw)
    return [(w / total, a, b) for w, a, b in raw]


def _sample_bw_mass(
    m_pole: float,
    gamma: float,
    m_min: float,
    m_max: float,
    rng: np.random.Generator,
) -> float:
    """Sample a mass from the q² Breit–Wigner, truncated to [m_min, m_max]."""
    if m_max <= m_min:
        raise ValueError("empty Breit–Wigner window")
    g = m_pole * gamma
    pole2 = m_pole * m_pole

    def cdf(q2: float) -> float:
        return math.atan((q2 - pole2) / g) / math.pi + 0.5

    lo = cdf(m_min * m_min)
    hi = cdf(m_max * m_max)
    u = float(rng.uniform(lo, hi))
    # Stay off the arctan branch cut of the inverse.
    u = min(max(u, 1.0e-12), 1.0 - 1.0e-12)
    q2 = pole2 + g * math.tan(math.pi * (u - 0.5))
    q2 = min(max(q2, m_min * m_min), m_max * m_max)
    return math.sqrt(q2)


_VIRTUAL_GRID: dict[tuple[float, float, float, float], tuple[np.ndarray, np.ndarray, float]] = {}


def sample_virtual_pair(
    m_parent: float,
    m_pole: float,
    gamma: float,
    rng: np.random.Generator,
    m_min: float = 0.30,
) -> tuple[float, float]:
    """Sample two virtual-boson masses below ``m_parent``.

    The joint density is a product of q² Breit–Wigners times two-body phase
    space and the leading H → VV tensor factor (λ + 12 m1² m2²).
    """
    if m_parent <= 2.0 * m_min:
        raise RuntimeError("parent is too light to decay to two bosons")
    key = (round(m_parent, 4), round(m_pole, 4), round(gamma, 4), round(m_min, 4))
    cached = _VIRTUAL_GRID.get(key)
    if cached is None:
        n = 140
        masses = np.linspace(m_min, m_parent - m_min, n)
        m1, m2 = np.meshgrid(masses, masses, indexing="ij")
        q1 = m1 * m1
        q2 = m2 * m2
        pole2 = m_pole * m_pole
        breadth = (m_pole * gamma) ** 2
        bw = 1.0 / ((q1 - pole2) ** 2 + breadth) / ((q2 - pole2) ** 2 + breadth)
        s = m_parent * m_parent
        lam = s * s + q1 * q1 + q2 * q2 - 2.0 * s * q1 - 2.0 * s * q2 - 2.0 * q1 * q2
        ok = (m1 + m2 < m_parent - 0.05) & (lam > 0.0)
        ps = np.sqrt(np.maximum(lam, 0.0)) * (np.maximum(lam, 0.0) + 12.0 * q1 * q2)
        weight = np.where(ok, bw * ps, 0.0).ravel()
        total = float(weight.sum())
        if total <= 0.0:
            raise RuntimeError("virtual-boson mass sampling failed")
        cached = (masses, np.cumsum(weight), total)
        _VIRTUAL_GRID[key] = cached
    masses, cdf, total = cached
    u = float(rng.random()) * total
    index = int(np.searchsorted(cdf, u, side="left"))
    index = min(index, cdf.size - 1)
    i, j = divmod(index, masses.size)
    return float(masses[i]), float(masses[j])


def michel_x(rng: np.random.Generator) -> float:
    """Sample x = 2E/m from the massless Michel spectrum 2 x² (3 - 2x)."""
    while True:
        x = float(rng.random())
        if float(rng.random()) * 2.0 <= 2.0 * x * x * (3.0 - 2.0 * x):
            return x


def _massless_pair_in_rest(
    pair: FourVector,
    rng: np.random.Generator,
) -> tuple[FourVector, FourVector]:
    """Split a timelike 4-vector into two massless back-to-back daughters."""
    m = pair.mass
    if m <= 0.0:
        zero = FourVector(0.0, 0.0, 0.0, 0.0)
        return zero, zero
    cos_theta, phi = _angles(rng)
    half = 0.5 * m
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    px = half * sin_theta * math.cos(phi)
    py = half * sin_theta * math.sin(phi)
    pz = half * cos_theta
    a = FourVector(half, px, py, pz)
    b = FourVector(half, -px, -py, -pz)
    return boost_from_rest(a, pair), boost_from_rest(b, pair)


def leptonic_decay(
    parent: FourVector,
    pdg_lepton: int,
    pdg_nu_bar: int,
    pdg_nu_parent: int,
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    """μ or τ → ℓ ν ν with the Michel energy spectrum. Momenta sum to ``parent``."""
    m = parent.mass
    m_l = species(pdg_lepton).mass
    for _ in range(1000):
        x = michel_x(rng)
        energy = 0.5 * x * m
        if energy <= m_l:
            continue
        p_mag = math.sqrt(energy * energy - m_l * m_l)
        cos_theta, phi = _angles(rng)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        px = p_mag * sin_theta * math.cos(phi)
        py = p_mag * sin_theta * math.sin(phi)
        pz = p_mag * cos_theta
        lepton_rest = FourVector(energy, px, py, pz)
        pair_rest = FourVector(m - energy, -px, -py, -pz)
        if pair_rest.m2 < -1.0e-8:
            continue
        nu_a, nu_b = _massless_pair_in_rest(pair_rest, rng)
        lepton = boost_from_rest(lepton_rest, parent)
        return [
            (pdg_lepton, lepton),
            (pdg_nu_bar, boost_from_rest(nu_a, parent)),
            (pdg_nu_parent, boost_from_rest(nu_b, parent)),
        ]
    raise RuntimeError("Michel decay sampling failed")


def _two(parent: FourVector, pdg1: int, pdg2: int, rng: np.random.Generator) -> list[tuple[int, FourVector]]:
    cos_theta, phi = _angles(rng)
    p1, p2 = two_body_decay(parent, species(pdg1).mass, species(pdg2).mass, cos_theta, phi)
    return [(pdg1, p1), (pdg2, p2)]


def _two_massive(
    parent: FourVector,
    pdg1: int,
    pdg2: int,
    m1: float,
    m2: float,
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    cos_theta, phi = _angles(rng)
    p1, p2 = two_body_decay(parent, m1, m2, cos_theta, phi)
    return [(pdg1, p1), (pdg2, p2)]


def _rest_frame_vector(mass: float, px: float, py: float, pz: float) -> FourVector:
    e = math.sqrt(mass * mass + px * px + py * py + pz * pz)
    return FourVector(e, px, py, pz)


def _sequential_pions(
    parent: FourVector,
    pion_pdgs: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    """Momentum-conserving sequential split. Used for the small tau remainder."""
    if len(pion_pdgs) == 2:
        return _two(parent, pion_pdgs[0], pion_pdgs[1], rng)
    first, *rest = pion_pdgs
    m_rest_min = sum(species(pdg).mass for pdg in rest)
    m_parent = parent.mass
    m_first = species(first).mass
    # Sample the recoil mass flat in mass-squared inside the allowed window.
    lo = m_rest_min
    hi = m_parent - m_first
    if hi <= lo:
        raise RuntimeError("pion system does not fit")
    m2 = float(rng.uniform(lo * lo, hi * hi))
    m_rest = math.sqrt(m2)
    cos_theta, phi = _angles(rng)
    p_first, p_rest = two_body_decay(parent, m_first, m_rest, cos_theta, phi)
    daughters = _sequential_pions(p_rest, rest, rng)
    return [(first, p_first), *daughters]


def _rho_decay(parent: FourVector, rho_pdg: int, rng: np.random.Generator) -> list[tuple[int, FourVector]]:
    if rho_pdg == 113:
        return _two(parent, 211, -211, rng)
    if rho_pdg == 213:
        return _two(parent, 211, 111, rng)
    if rho_pdg == -213:
        return _two(parent, -211, 111, rng)
    raise ValueError(f"not a rho: {rho_pdg}")


def _sample_resonance_mass(pole: float, gamma: float, m_min: float, m_max: float, rng: np.random.Generator) -> float:
    return _sample_bw_mass(pole, gamma, m_min, m_max, rng)


def decay_tau(parent: FourVector, pdg: int, rng: np.random.Generator) -> list[tuple[int, FourVector]]:
    """Decay a tau. ``pdg`` is +15 or -15; charged daughters are conjugated for τ+."""
    sign = 1 if pdg > 0 else -1
    # TAU_BR is written for τ-.
    names = [name for name, _br in TAU_BR]
    weights = [br for _name, br in TAU_BR]
    for _attempt in range(80):
        channel = names[_pick(weights, rng)]
        for _retry in range(40):
            try:
                return _decay_tau_channel(parent, sign, channel, rng)
            except (RuntimeError, ValueError):
                continue
    raise RuntimeError("tau decay failed")


def _decay_tau_channel(
    parent: FourVector,
    sign: int,
    channel: str,
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    # sign = +1 for τ- (pdg +15). Charged meson PDG ids below are for τ- and then flipped.
    nu = 16 if sign > 0 else -16
    if channel == "enu":
        lepton = 11 if sign > 0 else -11
        nu_bar = -12 if sign > 0 else 12
        return leptonic_decay(parent, lepton, nu_bar, nu, rng)
    if channel == "munu":
        lepton = 13 if sign > 0 else -13
        nu_bar = -14 if sign > 0 else 14
        return leptonic_decay(parent, lepton, nu_bar, nu, rng)
    if channel == "pi":
        pion = -211 if sign > 0 else 211
        return _two(parent, pion, nu, rng)
    if channel == "rho":
        return _tau_rho(parent, sign, nu, rng)
    if channel == "a1_neutral":
        return _tau_a1(parent, sign, nu, charged_mode=False, rng=rng)
    if channel == "a1_charged":
        return _tau_a1(parent, sign, nu, charged_mode=True, rng=rng)
    if channel in ("four_pi", "rest"):
        # τ- → π- π0 π0 π0 ν, a stand-in that keeps charge and 4-momentum exact.
        pions = [-211 if sign > 0 else 211, 111, 111, 111]
        return _tau_with_pions(parent, nu, pions, rng)
    raise ValueError(channel)


def _tau_with_pions(
    parent: FourVector,
    nu_pdg: int,
    pion_pdgs: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    m_hadrons = sum(species(pdg).mass for pdg in pion_pdgs)
    m_tau = parent.mass
    if m_tau <= m_hadrons:
        raise RuntimeError("tau is below the hadronic threshold")
    lo = m_hadrons
    hi = m_tau
    m2 = float(rng.uniform(lo * lo, hi * hi))
    m_sys = math.sqrt(m2)
    cos_theta, phi = _angles(rng)
    p_sys, p_nu = two_body_decay(parent, m_sys, 0.0, cos_theta, phi)
    pions = _sequential_pions(p_sys, pion_pdgs, rng)
    return [(nu_pdg, p_nu), *pions]


def _tau_rho(parent: FourVector, sign: int, nu: int, rng: np.random.Generator) -> list[tuple[int, FourVector]]:
    m_min = species(211).mass + species(111).mass
    m_max = parent.mass
    m_rho = _sample_resonance_mass(M_RHO, GAMMA_RHO, m_min, m_max, rng)
    cos_theta, phi = _angles(rng)
    rho_pdg = -213 if sign > 0 else 213
    p_rho, p_nu = two_body_decay(parent, m_rho, 0.0, cos_theta, phi)
    pions = _rho_decay(p_rho, rho_pdg, rng)
    return [(nu, p_nu), *pions]


def _tau_a1(
    parent: FourVector,
    sign: int,
    nu: int,
    charged_mode: bool,
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    # a1 mass must leave room for a neutrino and for a rho+pion inside the a1.
    m_pi = species(211).mass
    m_min = M_RHO * 0.4 + m_pi
    m_max = parent.mass - 1.0e-3
    if m_max <= m_min:
        raise RuntimeError("no a1 window")
    m_a1 = _sample_resonance_mass(M_A1, GAMMA_A1, m_min, m_max, rng)
    cos_theta, phi = _angles(rng)
    p_a1, p_nu = two_body_decay(parent, m_a1, 0.0, cos_theta, phi)
    # Decay a1 → ρ π, then ρ → ππ. Masses have to fit.
    if charged_mode:
        # a1- → ρ0 π- → π+ π- π-
        rho_pdg = 113
        bachelor = -211 if sign > 0 else 211
        m_rho_min = 2.0 * m_pi
    else:
        # a1- → ρ- π0 → π- π0 π0
        rho_pdg = -213 if sign > 0 else 213
        bachelor = 111
        m_rho_min = m_pi + species(111).mass
    m_rho_max = m_a1 - species(bachelor).mass - 1.0e-4
    if m_rho_max <= m_rho_min:
        raise RuntimeError("a1 below rho pi threshold")
    m_rho = _sample_resonance_mass(M_RHO, GAMMA_RHO, m_rho_min, m_rho_max, rng)
    cos2, phi2 = _angles(rng)
    p_rho, p_bach = two_body_decay(p_a1, m_rho, species(bachelor).mass, cos2, phi2)
    pions = _rho_decay(p_rho, rho_pdg, rng)
    return [(nu, p_nu), (bachelor, p_bach), *pions]


def decay_once(
    pdg: int,
    parent: FourVector,
    rng: np.random.Generator,
    *,
    force_muon: bool,
) -> list[tuple[int, FourVector]] | None:
    """Return lab-frame daughters, or None if this species is treated as stable.

    The returned 4-momenta sum to ``parent``.
    """
    a = abs(pdg)
    if a == 13 and not force_muon:
        return None
    if a == 13 and force_muon:
        # μ- → e- ν̄e νμ. Conjugate for μ+.
        if pdg > 0:
            return leptonic_decay(parent, 11, -12, 14, rng)
        return leptonic_decay(parent, -11, 12, -14, rng)
    if a == 15:
        return decay_tau(parent, pdg, rng)
    if a == 6:
        # t → W+ b, t̄ → W- b̄. Essentially the only channel.
        if pdg > 0:
            return _two(parent, 24, 5, rng)
        return _two(parent, -24, -5, rng)
    if a == 24:
        table = w_channel_weights(parent.mass)
        if not table:
            raise RuntimeError("W has no open channel at this mass")
        idx = _pick([w for w, _a, _b in table], rng)
        _w, d1, d2 = table[idx]
        if pdg < 0:
            d1, d2 = -d1, -d2
        return _two(parent, d1, d2, rng)
    if pdg == 23:
        table = z_channel_weights(parent.mass)
        if not table:
            raise RuntimeError("Z has no open channel at this mass")
        idx = _pick([w for w, _a, _b in table], rng)
        _w, d1, d2 = table[idx]
        return _two(parent, d1, d2, rng)
    if pdg == 25:
        return _decay_higgs(parent, rng)
    if pdg == 111:
        return _two(parent, 22, 22, rng)
    if a == 113 or a == 213:
        return _rho_decay(parent, pdg, rng)
    if a == 20213:
        # Not produced except inside tau decays, which expand the a1 themselves.
        return None
    if a in (11, 12, 14, 16, 22, 21, 1, 2, 3, 4, 5, 211):
        return None
    if is_quark(pdg):
        return None
    return None


def _decay_higgs(parent: FourVector, rng: np.random.Generator) -> list[tuple[int, FourVector]]:
    names = [name for name, _br in HIGGS_BR]
    weights = [br for _name, br in HIGGS_BR]
    channel = names[_pick(weights, rng)]
    if channel == "bb":
        return _two(parent, 5, -5, rng)
    if channel == "cc":
        return _two(parent, 4, -4, rng)
    if channel == "tautau":
        return _two(parent, 15, -15, rng)
    if channel == "mumu":
        return _two(parent, 13, -13, rng)
    if channel == "gammagamma":
        return _two(parent, 22, 22, rng)
    if channel == "gg":
        return _two(parent, 21, 21, rng)
    if channel == "zgamma":
        return _two(parent, 23, 22, rng)
    if channel == "ww":
        m1, m2 = sample_virtual_pair(parent.mass, M_W, GAMMA_W, rng)
        return _two_massive(parent, 24, -24, m1, m2, rng)
    if channel == "zz":
        m1, m2 = sample_virtual_pair(parent.mass, M_Z, GAMMA_Z, rng, m_min=0.5)
        return _two_massive(parent, 23, 23, m1, m2, rng)
    raise ValueError(channel)

