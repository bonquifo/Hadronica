"""CODATA and PDG 2024 inputs used by the generator.

Masses and widths are the PDG 2024 listings (S. Navas et al., Phys. Rev. D
110, 030001 (2024)), in GeV. The electromagnetic coupling is the CODATA
low-energy value. The leptonic effective weak mixing angle is the PDG
effective angle used for improved-Born couplings.
"""

from __future__ import annotations

import math

# Defined speed of light, and CODATA ħ in GeV·s (6.582119569×10⁻¹⁶ eV·s).
C_M_PER_S = 299_792_458.0
HBAR_GEV_S = 6.582119569e-25

# PDG 2024 π⁰ mean life, 8.43×10⁻¹⁷ s.
TAU_PI0_S = 8.43e-17

# Charged leptons
M_E = 0.51099895000e-3
M_MU = 0.1056583755
M_TAU = 1.77693
GAMMA_MU = 2.9959836e-19  # essentially stable on a detector scale
# tau width from the PDG lifetime: c tau = 87.03 um
C_TAU_TAU_M = 87.03e-6
C_TAU_MU_M = 658.6384
C_TAU_PI_M = 7.8045

# Quarks. Light-quark entries are MS-bar masses; the top entry is the
# PDG direct measurement used as the kinematic mass.
M_U = 2.16e-3
M_D = 4.70e-3
M_S = 93.5e-3
M_C = 1.273
M_B = 4.183
M_T = 172.57
GAMMA_T = 1.42

# Gauge and Higgs bosons
M_W = 80.3692
GAMMA_W = 2.085
M_Z = 91.1880
GAMMA_Z = 2.4955
M_H = 125.20
# SM prediction quoted by the PDG Higgs review for m_H = 125 GeV.
# The PDG measured width is 3.7 +1.9 -1.4 MeV; decays use branching
# fractions, so this width is informational and used only as a linewidth.
GAMMA_H = 4.07e-3

# Mesons that appear as decay products
M_PI = 0.13957039
M_PI0 = 0.1349768
M_RHO = 0.77526
GAMMA_RHO = 0.1491
M_A1 = 1.230
GAMMA_A1 = 0.420

# Couplings
ALPHA_INV = 137.035999084  # CODATA alpha^{-1}(0)
ALPHA = 1.0 / ALPHA_INV
G_F = 1.1663788e-5  # GeV^-2
SIN2_THETA_W = 0.23153  # effective leptonic sin^2 theta_W
ALPHA_S_MZ = 0.1180

# 1 GeV^-2 = 0.389379292 mb = 3.89379292e8 pb
PB_PER_GEV2 = 3.89379292e8

# p[GeV/c] = 0.299792458 * q * B[T] * R[m]
P_OVER_QRB = 0.299792458
B_SOLENOID_T = 3.8  # CMS-like barrel field

# |V_ij| for the first two CKM rows (PDG 2024 magnitudes).
CKM_ABS = {
    (2, 1): 0.97373,
    (2, 3): 0.22431,
    (2, 5): 0.00382,
    (4, 1): 0.221,
    (4, 3): 0.975,
    (4, 5): 0.0408,
}

# Fiducial polar-angle cut for processes that diverge at theta = 0, pi.
THETA_FIDUCIAL = math.radians(10.0)

# Quark-pair production is a parton-level Born process. These floors keep
# it out of the exclusive light-hadron and bottomonium resonance regions.
QUARK_CONTINUUM_GEV = {
    1: 5.0,
    2: 5.0,
    3: 5.0,
    4: 5.0,
    5: 12.0,
    6: 0.0,
}


def format_cross_section(sigma_pb: float) -> str:
    """Format a cross section in pb using nb / pb / fb / ab."""
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
