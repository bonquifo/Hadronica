"""Physical inputs used by the generator, with their sources.

Unless noted otherwise, masses, widths, lifetimes, and branching fractions are
the 2026 Review of Particle Physics summary tables:
    F. Takahashi et al. (Particle Data Group), Int. J. Mod. Phys. A 41, 2630011 (2026),
    https://pdg.lbl.gov/2026/
Fundamental constants are CODATA 2022 as tabulated by PDG 2026 (Table 1.1),
except the Fermi constant, which is the PDG 2026 value. Energies and masses are in GeV.
"""

from __future__ import annotations

import math

REFERENCE = "PDG 2026, Int. J. Mod. Phys. A 41, 2630011 (2026); CODATA 2022"

# Exact SI constants. ħ = 6.582119569... × 10⁻²² MeV s.
C_M_PER_S = 299_792_458.0
HBAR_GEV_S = 6.582119569e-25

# π⁰ mean life, (8.43 ± 0.13) × 10⁻¹⁷ s.
TAU_PI0_S = 8.43e-17

# Charged leptons
M_E = 0.51099895069e-3  # (16) × 10⁻¹¹ MeV, CODATA 2022
M_MU = 0.1056583755  # ± 2.3 × 10⁻⁹ GeV
M_TAU = 1.77693  # ± 0.09 MeV
GAMMA_MU = HBAR_GEV_S / 2.1969811e-6  # from τ_μ = 2.1969811(22) μs
C_TAU_TAU_M = 87.03e-6  # τ_τ = 290.3 fs
C_TAU_MU_M = 658.6384
C_TAU_PI_M = 7.8045  # τ_π± = 2.6033 × 10⁻⁸ s

# Quarks. u, d, s are MS-bar masses at 2 GeV; c and b are m(m) in MS-bar;
# the top entry is the PDG average of direct measurements, used as the
# kinematic (pole-like) mass.
M_U = 2.16e-3
M_D = 4.70e-3
M_S = 92.9e-3
M_C = 1.2729
M_B = 4.186
M_T = 172.60  # ± 0.27 GeV
GAMMA_T = 1.42  # +0.19 −0.15 GeV

# Gauge and Higgs bosons
M_W = 80.3625  # ± 0.0077 GeV (PDG average; excludes CDF 2022)
GAMMA_W = 2.14  # ± 0.05 GeV
M_Z = 91.1879  # ± 0.0020 GeV
GAMMA_Z = 2.4955  # ± 0.0023 GeV
M_H = 125.13  # ± 0.11 GeV
# Standard Model total width from the LHC Higgs Cross Section Working Group
# (CERN Yellow Report 4, CERN-2017-002-M, arXiv:1610.07922) at m_H = 125.10 GeV,
# the tabulated mass closest to the PDG value. The PDG measured width is
# 3.0 +1.5 −0.7 MeV; decays use branching fractions, so this width is used only
# for the lifetime ħ/Γ.
GAMMA_H = 4.101e-3

# Mesons that appear as tau decay products
M_PI = 0.13957039
M_PI0 = 0.1349768
M_RHO0 = 0.77526  # ρ(770)⁰ Breit–Wigner mass
GAMMA_RHO0 = 0.1474
M_RHO = 0.77511  # ρ(770)± Breit–Wigner mass
GAMMA_RHO = 0.1491
M_A1 = 1.230  # a1(1260) Breit–Wigner mass, ± 40 MeV
GAMMA_A1 = 0.425  # PDG lists the width only as "250 to 600 MeV"; this is the midpoint

# Couplings
ALPHA_INV = 137.035999177  # CODATA 2022 α⁻¹(0)
ALPHA = 1.0 / ALPHA_INV
G_F = 1.1663785e-5  # G_F/(ħc)³ in GeV⁻², ± 6 × 10⁻¹²: PDG 2026 (CODATA 2022 gives 1.1663787e-5)
SIN2_THETA_W = 0.23154  # effective leptonic sin²θ_eff, SM fit (PDG Electroweak review)
ALPHA_S_MZ = 0.1180  # ± 0.0009, PDG world average
# Five-flavor hadronic vacuum polarization at the Z, the value used in the
# PDG Electroweak review fit (Davier et al., Eur. Phys. J. C 80 (2020) 241).
DALPHA_HAD5_MZ = 0.02760

# (ħc)² = 0.3893793721 GeV² mb, exact, so 1 GeV⁻² = 3.893793721 × 10⁸ pb.
PB_PER_GEV2 = 3.893793721e8

# p[GeV/c] = 0.299792458 * q * B[T] * R[m]
P_OVER_QRB = 0.299792458
# CMS solenoid operating field. The coil is 12.5 m long with a 6 m free bore
# (CMS Collaboration; V. Klyukhin et al., arXiv:2201.07557).
B_SOLENOID_T = 3.8

# |V_ij| for the first two CKM rows: PDG 2026 CKM review, averages of the
# individual direct measurements (eqs. 12.7–12.12), not a unitarity fit.
CKM_ABS = {
    (2, 1): 0.97367,
    (2, 3): 0.22431,
    (2, 5): 0.00389,
    (4, 1): 0.2247,
    (4, 3): 0.969,
    (4, 5): 0.0407,
}

# Fiducial polar-angle cut for processes that diverge at theta = 0, pi.
THETA_FIDUCIAL = math.radians(10.0)

# Quark-pair production is a parton-level Born process. These floors keep
# it out of the exclusive light-hadron, charmonium, and bottomonium
# resonance regions, where a free-quark formula does not apply.
QUARK_CONTINUUM_GEV = {
    1: 5.0,
    2: 5.0,
    3: 5.0,
    4: 5.0,
    5: 12.0,
    6: 0.0,
}


def format_cross_section(sigma_pb: float) -> str:
    """Format a cross section in pb using μb / nb / pb / fb / ab."""
    if sigma_pb <= 0.0 or not math.isfinite(sigma_pb):
        return "0"
    a = sigma_pb
    if a >= 1.0e6:
        return f"{a / 1.0e6:.3g} μb"
    if a >= 1.0e3:
        return f"{a / 1.0e3:.3g} nb"
    if a >= 1.0:
        return f"{a:.3g} pb"
    if a >= 1.0e-3:
        return f"{a * 1.0e3:.3g} fb"
    return f"{a * 1.0e6:.3g} ab"


def format_energy(gev: float) -> str:
    if gev < 1.0:
        return f"{gev * 1.0e3:.2f} MeV"
    if gev < 10.0:
        return f"{gev:.3f} GeV"
    return f"{gev:.2f} GeV"
