"""Leading-order electroweak cross sections.

Fermion pairs use the improved Born approximation (IBA): photon exchange with
the running coupling α(s), Z exchange normalized with G_F, and their
interference: the Born-type structure with effective couplings of the LEP/SLD
Z-pole analyses (S. Schael et al., Phys. Rept. 427, 257 (2006), §1.5).
The reduced Z amplitude is

    χ(s) = [G_F M_Z² / (2 √2 π α(s))] s / (s - M_Z² + i s Γ_Z / M_Z),

with g_A = T_3 and g_V = T_3 - 2 Q sin²θ_eff (PDG Electroweak review).
At the Z pole the resonant term equals the Breit–Wigner built from the same
partial widths. Mass thresholds suppress the vector piece by β(3-β²)/2, the
axial piece by β³, and the vector–axial (forward–backward) piece by β².

α(s) = α(0) / (1 - Δα_lep(s) - Δα_had(s)). The lepton loops are the one-loop
leading-log sum; the hadronic part is the PDG value of Δα_had^(5)(M_Z²) run
with the perturbative five-flavor quark loop (approximate below ~10 GeV).

e⁺e⁻ → ν_e ν̄_e (and μ⁺μ⁻ → ν_μ ν̄_μ) add the t-channel W exchange to the
s-channel Z. After a Fierz rearrangement the W only feeds the left-handed
beam amplitude, g_L → g_L P_Z(s) + P_W(t), the same "+1" that appears in
ν_e e scattering (PDG Electroweak review, neutrino–electron scattering).

Higgsstrahlung uses the Born formula (A. Djouadi, Phys. Rept. 457 (2008) 1,
arXiv:hep-ph/0503172; W. Kilian, M. Krämer, P. M. Zerwas,
Phys. Lett. B 373 (1996) 135, arXiv:hep-ph/9512355),

    σ(ZH) = [G_F² M_Z⁴ / (96 π s)] (v_e² + a_e²) λ^½ (λ + 12 M_Z²/s) / (1 - M_Z²/s)²,

with a_e = -1 and v_e = -1 + 4 sin²θ_eff. It is a Born (tree-level) result.

Bhabha scattering and e+e- → γγ are QED Born results with a 10° fiducial cut.
The Z diagrams of Bhabha are not included.
"""

from __future__ import annotations

import math

import numpy as np

from smlab.constants import (
    ALPHA,
    ALPHA_S_MZ,
    DALPHA_HAD5_MZ,
    G_F,
    GAMMA_Z,
    M_B,
    M_C,
    M_E,
    M_MU,
    M_T,
    M_TAU,
    M_W,
    M_Z,
    PB_PER_GEV2,
    SIN2_THETA_W,
    THETA_FIDUCIAL,
)
from smlab.kinematics import kallen, velocity_parameter
from smlab.particles import color_count, species, weak_isospin


def alpha_em(s: float) -> float:
    """Running QED coupling α(s) for a timelike s > 0 (real part of the vacuum polarization).

    Leptons: (α/3π) Σ_l [ln(s/m_l²) - 5/3], the one-loop leading-log form,
    dropped where it would turn negative (s near or below that lepton's pair
    threshold). Hadrons: Δα_had^(5)(M_Z²) from the PDG, moved to other scales
    with the perturbative quark loop (α/3π) (11/3)(1 + α_s/π) ln(s/M_Z²) and
    floored at zero.
    """
    if s <= 0.0:
        return ALPHA
    k = ALPHA / (3.0 * math.pi)
    d_lep = 0.0
    for mass in (M_E, M_MU, M_TAU):
        d_lep += max(0.0, math.log(s / (mass * mass)) - 5.0 / 3.0)
    d_lep *= k
    r_had = (11.0 / 3.0) * (1.0 + ALPHA_S_MZ / math.pi)
    d_had = max(0.0, DALPHA_HAD5_MZ + k * r_had * math.log(s / (M_Z * M_Z)))
    return ALPHA / (1.0 - d_lep - d_had)


def chi_z(s: float, width: float = GAMMA_Z, alpha: float | None = None) -> complex:
    """Improved-Born Z propagator factor, G_F scheme, s-dependent width.

    ``alpha`` defaults to the running α(s) that also multiplies the photon term,
    so the pure-Z term is independent of α and the interference scales as α(s).
    """
    if s <= 0.0:
        return 0.0j
    a = alpha_em(s) if alpha is None else alpha
    pref = (G_F * M_Z * M_Z) / (2.0 * math.sqrt(2.0) * math.pi * a)
    denom = (s - M_Z * M_Z) + 1j * (s * width / M_Z)
    return pref * s / denom


def couplings(pdg: int) -> tuple[float, float, float]:
    """Return (Q, g_V, g_A) for the fermion. Antiparticle ids use the fermion's couplings."""
    particle = species(abs(pdg))
    q = particle.charge
    t3 = weak_isospin(pdg)
    g_a = t3
    g_v = t3 - 2.0 * q * SIN2_THETA_W
    return q, g_v, g_a


def _b0(n_f: int) -> float:
    return (11.0 - 2.0 * n_f / 3.0) / (4.0 * math.pi)


def alpha_s(scale_gev: float) -> float:
    """One-loop α_s(Q), fixed to the PDG α_s(M_Z) with five active flavors.

    The number of active flavors changes at m_c, m_b, and m_t, and α_s is
    continuous across each threshold (leading-order matching).
    """
    q = max(scale_gev, 1.0)
    inv = 1.0 / ALPHA_S_MZ
    mu = M_Z
    n_f = 5
    if q > M_T:
        inv += _b0(5) * math.log((M_T * M_T) / (mu * mu))
        mu, n_f = M_T, 6
    else:
        for threshold, below in ((M_B, 4), (M_C, 3)):
            if q < threshold:
                inv += _b0(n_f) * math.log((threshold * threshold) / (mu * mu))
                mu, n_f = threshold, below
    inv += _b0(n_f) * math.log((q * q) / (mu * mu))
    if inv < 0.5:
        return 2.0
    return 1.0 / inv


def qcd_factor(final_pdg: int, sqrt_s: float) -> float:
    """Final-state QCD correction 1 + α_s(s)/π for quarks, else 1."""
    if color_count(final_pdg) == 1:
        return 1.0
    return 1.0 + alpha_s(sqrt_s) / math.pi


def _threshold_factors(beta: float) -> tuple[float, float]:
    if beta <= 0.0:
        return 0.0, 0.0
    r_v = beta * (3.0 - beta * beta) / 2.0
    r_a = beta**3
    return r_v, r_a


def fermion_pair_amplitudes(
    sqrt_s: float,
    pdg_i: int,
    pdg_f: int,
    width: float = GAMMA_Z,
) -> tuple[float, float, float, float]:
    """Return (A_V, A_A, A_1, β) before color and QCD.

    The angular density in dσ/dΩ = (α² / 4s) f(cos θ) is

        f = A_V β (2 - β² + β² cos²θ) + A_A β³ (1 + cos²θ) + A_1 β² cosθ,

    which reduces to A_0 (1 + cos²θ) + A_1 cosθ when β → 1, and whose integral
    is the standard total rate with vector factor β(3-β²)/2 and axial factor β³.
    The vector–axial interference behind the asymmetry carries one power of β
    from phase space and one from the final-state current, hence β².
    χ uses the running α(s).
    """
    s = sqrt_s * sqrt_s
    mass = species(abs(pdg_f)).mass
    beta = velocity_parameter(sqrt_s, mass)
    if beta <= 0.0 or s <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    q_i, g_vi, g_ai = couplings(pdg_i)
    q_f, g_vf, g_af = couplings(pdg_f)
    chi = chi_z(s, width, alpha_em(s))
    re_chi = chi.real
    chi2 = abs(chi) ** 2
    qprod = q_i * q_f
    initial = g_vi * g_vi + g_ai * g_ai
    a_v = (
        qprod * qprod
        + 2.0 * qprod * g_vi * g_vf * re_chi
        + initial * (g_vf * g_vf) * chi2
    )
    a_a = initial * (g_af * g_af) * chi2
    a_1 = (
        4.0 * qprod * g_ai * g_af * re_chi
        + 8.0 * g_vi * g_ai * g_vf * g_af * chi2
    )
    return a_v, a_a, a_1, beta


def fermion_pair_sigma(
    sqrt_s: float,
    pdg_i: int,
    pdg_f: int,
    width: float = GAMMA_Z,
) -> float:
    """Improved-Born cross section in GeV^-2 for f_i f̄_i → f_f f̄_f, s-channel only.

    Includes N_c,final / N_c,initial and the final-state factor 1 + α_s/π
    (the massless-quark O(α_s) correction).
    """
    a_v, a_a, _a1, beta = fermion_pair_amplitudes(sqrt_s, pdg_i, pdg_f, width)
    if beta <= 0.0:
        return 0.0
    r_v, r_a = _threshold_factors(beta)
    a_eff = a_v * r_v + a_a * r_a
    if a_eff <= 0.0:
        return 0.0
    s = sqrt_s * sqrt_s
    color = color_count(pdg_f) / color_count(pdg_i)
    alpha = alpha_em(s)
    born = (4.0 * math.pi * alpha * alpha / (3.0 * s)) * a_eff
    return color * qcd_factor(pdg_f, sqrt_s) * born


def fermion_angular_density(cos_theta: float, a_v: float, a_a: float, a_1: float, beta: float) -> float:
    """Un-normalized dσ/dΩ density. The polar angle is the outgoing fermion's."""
    c2 = cos_theta * cos_theta
    return (
        a_v * beta * (2.0 - beta * beta + beta * beta * c2)
        + a_a * (beta**3) * (1.0 + c2)
        + a_1 * beta * beta * cos_theta
    )


def partial_width(
    pdg: int,
    mass: float = M_Z,
    width_scale_qcd: bool = True,
) -> float:
    """Leading-order partial width of a Z-like vector of the given mass, in GeV.

    Γ = N_c G_F m³ / (6 π √2) × (g_V² R_V + g_A² R_A) × (1 + α_s/π for quarks).
    """
    fermion = species(abs(pdg))
    if mass <= 2.0 * fermion.mass:
        return 0.0
    _q, g_v, g_a = couplings(pdg)
    beta = velocity_parameter(mass, fermion.mass)
    r_v, r_a = _threshold_factors(beta)
    coeff = G_F * mass**3 / (6.0 * math.pi * math.sqrt(2.0))
    qcd = qcd_factor(pdg, mass) if width_scale_qcd else 1.0
    return coeff * color_count(pdg) * (g_v * g_v * r_v + g_a * g_a * r_a) * qcd


def z_partial_widths(mass: float = M_Z) -> dict[int, float]:
    """Partial widths keyed by the fermion PDG id (the particle, not the anti)."""
    flavors = (12, 14, 16, 11, 13, 15, 2, 4, 1, 3, 5)
    return {pdg: partial_width(pdg, mass) for pdg in flavors if partial_width(pdg, mass) > 0.0}


# ---------------------------------------------------------------------------
# Same-generation neutrino pairs: s-channel Z plus t-channel W
# ---------------------------------------------------------------------------

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def neutrino_w_dsigma_domega(sqrt_s: float, cos_theta: float) -> float:
    """dσ/dΩ in GeV^-2 for ℓ⁻ℓ⁺ → ν_ℓ ν̄_ℓ with Z and W exchange, massless.

    θ is the angle between the incoming ℓ⁻ and the outgoing ν_ℓ. With
    P_Z = M_Z² / (M_Z² - s - i s Γ_Z/M_Z) and P_W = M_W² / (M_W² - t),

        dσ/dΩ = (G_F² s / 32π²) [ |g_L P_Z + P_W|² (1+cosθ)² + g_R² |P_Z|² (1-cosθ)² ],

    g_L = -1/2 + sin²θ_eff, g_R = sin²θ_eff. Without P_W this is exactly the
    improved-Born Z term used for the other neutrino flavors.
    """
    s = sqrt_s * sqrt_s
    if s <= 0.0:
        return 0.0
    c = min(1.0, max(-1.0, cos_theta))
    g_l = -0.5 + SIN2_THETA_W
    g_r = SIN2_THETA_W
    p_z = (M_Z * M_Z) / (M_Z * M_Z - s - 1j * s * GAMMA_Z / M_Z)
    t = -0.5 * s * (1.0 - c)
    p_w = (M_W * M_W) / (M_W * M_W - t)
    left = abs(g_l * p_z + p_w) ** 2 * (1.0 + c) ** 2
    right = (g_r * g_r) * abs(p_z) ** 2 * (1.0 - c) ** 2
    return (G_F * G_F * s / (32.0 * math.pi * math.pi)) * (left + right)


def neutrino_w_sigma(sqrt_s: float) -> float:
    """Total σ(ℓ⁻ℓ⁺ → ν_ℓ ν̄_ℓ) in GeV^-2, Gauss–Legendre over the full solid angle."""
    total = 0.0
    for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
        total += weight * neutrino_w_dsigma_domega(sqrt_s, float(node))
    return 2.0 * math.pi * total


# ---------------------------------------------------------------------------
# Higgsstrahlung
# ---------------------------------------------------------------------------

def _zh_lambda(s: float, m_h: float, m_z: float) -> float:
    """Reduced Källén function λ/s². Negative below threshold."""
    return kallen(s, m_h * m_h, m_z * m_z) / (s * s)


def higgsstrahlung_sigma(sqrt_s: float, m_h: float | None = None, m_z: float | None = None) -> float:
    """Born σ(e+e- → ZH) in GeV^-2. Zero below the ZH threshold."""
    from smlab.constants import M_H

    m_h = M_H if m_h is None else m_h
    m_z = M_Z if m_z is None else m_z
    if sqrt_s <= m_h + m_z:
        return 0.0
    s = sqrt_s * sqrt_s
    lam = _zh_lambda(s, m_h, m_z)
    if lam <= 0.0:
        return 0.0
    beta = math.sqrt(lam)
    r = (m_z * m_z) / s
    # a_e = -1, v_e = -1 + 4 sin²θ, so v² + a² = 4 (g_V² + g_A²).
    v_e = -1.0 + 4.0 * SIN2_THETA_W
    a_e = -1.0
    va = v_e * v_e + a_e * a_e
    pref = (G_F * G_F * m_z**4) / (96.0 * math.pi * s)
    shape = beta * (beta * beta + 12.0 * r) / (1.0 - r) ** 2
    return pref * va * shape


def higgsstrahlung_density(cos_theta: float, sqrt_s: float, m_h: float | None = None) -> float:
    """Angular weight of the Z, proportional to β² sin²θ + 8 M_Z²/s.

    Integrated against the prefactor below, this reproduces ``higgsstrahlung_sigma``.
    """
    from smlab.constants import M_H

    m_h = M_H if m_h is None else m_h
    s = sqrt_s * sqrt_s
    lam = _zh_lambda(s, m_h, M_Z)
    if lam <= 0.0:
        return 0.0
    beta2 = lam
    r = (M_Z * M_Z) / s
    sin2 = 1.0 - cos_theta * cos_theta
    return beta2 * sin2 + 8.0 * r


# ---------------------------------------------------------------------------
# QED: Bhabha and two-photon annihilation
# ---------------------------------------------------------------------------

def _mandelstam(s: float, cos_theta: float) -> tuple[float, float]:
    # Massless CM relations. t and u are negative for physical angles.
    t = -0.5 * s * (1.0 - cos_theta)
    u = -0.5 * s * (1.0 + cos_theta)
    return t, u


def bhabha_dsigma_domega(sqrt_s: float, cos_theta: float) -> float:
    """QED Bhabha dσ/dΩ in GeV^-2, massless electrons, photon exchange only.

    dσ/dΩ = (α² / 2s) [(s²+u²)/t² + (t²+u²)/s² + 2u²/(s t)].
    θ is the outgoing electron angle relative to the incoming electron.
    """
    s = sqrt_s * sqrt_s
    if s <= 0.0:
        return 0.0
    c = min(1.0 - 1.0e-12, max(-1.0 + 1.0e-12, cos_theta))
    t, u = _mandelstam(s, c)
    bracket = (s * s + u * u) / (t * t) + (t * t + u * u) / (s * s) + (2.0 * u * u) / (s * t)
    return (ALPHA * ALPHA / (2.0 * s)) * bracket


def diphoton_dsigma_domega(sqrt_s: float, cos_theta: float) -> float:
    """QED e+e- → γγ dσ/dΩ in GeV^-2, identical-photon factor 1/2 included.

    dσ/dΩ = (α² / 2s) (1 + cos²θ) / sin²θ, integrated over the full fiducial solid angle.
    """
    s = sqrt_s * sqrt_s
    if s <= 0.0:
        return 0.0
    c = min(1.0 - 1.0e-12, max(-1.0 + 1.0e-12, cos_theta))
    sin2 = 1.0 - c * c
    return (ALPHA * ALPHA / (2.0 * s)) * (1.0 + c * c) / sin2


def _fiducial_cos() -> float:
    return math.cos(THETA_FIDUCIAL)


def integrate_dsigma_domega(sqrt_s: float, density, n: int = 4000) -> float:
    """∫ dΩ density over the fiducial polar range, uniform in cosθ with the Jacobian already in dσ/dΩ.

    dσ = ∫ (dσ/dΩ) 2π d(cosθ).
    """
    c_max = _fiducial_cos()
    # Simpson on [-c_max, c_max]. n must be even.
    if n % 2:
        n += 1
    step = (2.0 * c_max) / n
    total = 0.0
    for i in range(n + 1):
        c = -c_max + i * step
        w = 1.0 if i == 0 or i == n else (4.0 if i % 2 else 2.0)
        total += w * density(sqrt_s, c)
    integral_dc = total * step / 3.0
    return 2.0 * math.pi * integral_dc


def diphoton_sigma(sqrt_s: float) -> float:
    """Analytic fiducial σ(e+e- → γγ).

    With dσ/dΩ = (α² / 2s) (1+c²)/(1-c²) and |c| < c_max,
    σ = (2 π α² / s) [ln((1+c)/(1-c)) - c] evaluated at c_max.
    """
    s = sqrt_s * sqrt_s
    if s <= 0.0:
        return 0.0
    c = _fiducial_cos()
    bracket = math.log((1.0 + c) / (1.0 - c)) - c
    return (2.0 * math.pi * ALPHA * ALPHA / s) * bracket


def bhabha_sigma(sqrt_s: float) -> float:
    """Fiducial QED Bhabha cross section in GeV^-2."""
    s = sqrt_s * sqrt_s
    if s <= 0.0:
        return 0.0
    c_max = _fiducial_cos()
    c = np.linspace(-c_max, c_max, 4001)
    t = -0.5 * s * (1.0 - c)
    u = -0.5 * s * (1.0 + c)
    bracket = (s * s + u * u) / (t * t) + (t * t + u * u) / (s * s) + (2.0 * u * u) / (s * t)
    dsig = (ALPHA * ALPHA / (2.0 * s)) * bracket
    dc = np.diff(c)
    integral_dc = float(np.sum((dsig[1:] + dsig[:-1]) * 0.5 * dc))
    return 2.0 * math.pi * integral_dc


def to_pb(sigma_gev2: float) -> float:
    return sigma_gev2 * PB_PER_GEV2
