"""Physics tests for the Standard Model generator.

These lock the improved-Born identities, PDG-scale branching fractions,
4-momentum conservation, and the solenoid curvature convention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from smlab.constants import ALPHA, G_F, GAMMA_Z, M_H, M_MU, M_Z, PB_PER_GEV2, SIN2_THETA_W
from smlab.decays import (
    HIGGS_BR,
    TAU_BR,
    branching_table_w,
    branching_table_z,
    decay_tau,
    leptonic_decay,
    michel_x,
)
from smlab.electroweak import (
    alpha_em,
    alpha_s,
    bhabha_dsigma_domega,
    bhabha_sigma,
    chi_z,
    couplings,
    diphoton_dsigma_domega,
    diphoton_sigma,
    fermion_angular_density,
    fermion_pair_amplitudes,
    fermion_pair_sigma,
    higgsstrahlung_density,
    higgsstrahlung_sigma,
    integrate_dsigma_domega,
    neutrino_w_dsigma_domega,
    neutrino_w_sigma,
    partial_width,
    to_pb,
    z_partial_widths,
)
from smlab.generator import generate_event, sigma_pb
from smlab.kinematics import kallen, two_body_cm, two_body_decay, two_body_momentum
from smlab.lorentz import FourVector, boost, boost_from_rest
from smlab.particles import species
from smlab.processes import BEAMS, process_by_id
from smlab.tracks import curvature_radius, track_polyline


def _trap(ys: np.ndarray, xs: np.ndarray) -> float:
    return float(np.sum((ys[1:] + ys[:-1]) * 0.5 * np.diff(xs)))


def test_four_vector_mass_and_boost_roundtrip():
    parent = FourVector(10.0, 1.0, -2.0, 3.0)
    rest = FourVector(parent.mass, 0.0, 0.0, 0.0)
    boosted = boost_from_rest(rest, parent)
    assert boosted.almost_equal(parent, 1e-9)
    back = boost(boosted, tuple(-b for b in parent.beta()))
    assert back.almost_equal(rest, 1e-9)


def test_two_body_is_back_to_back_and_on_shell():
    p1, p2 = two_body_cm(91.188, M_MU, M_MU, 0.3, 0.4)
    total = p1 + p2
    assert total.almost_equal(FourVector(91.188, 0.0, 0.0, 0.0), 1e-8)
    assert p1.mass == pytest.approx(M_MU, rel=1e-10)
    assert p2.mass == pytest.approx(M_MU, rel=1e-10)
    expected = math.sqrt(kallen(91.188**2, M_MU**2, M_MU**2)) / (2 * 91.188)
    assert two_body_momentum(91.188, M_MU, M_MU) == pytest.approx(expected, rel=1e-12)


def test_decay_momenta_sum_to_parent():
    parent = FourVector(40.0, 5.0, -3.0, 12.0)
    a, b = two_body_decay(parent, 0.1, 0.2, -0.4, 1.2)
    assert (a + b).almost_equal(parent, 1e-8)


def test_alpha_s_at_the_z_is_the_pdg_input():
    assert alpha_s(M_Z) == pytest.approx(0.1180, rel=1e-12)


def test_z_couplings_and_partial_widths_match_pdg_fractions():
    _q, g_v, g_a = couplings(11)
    assert g_a == pytest.approx(-0.5)
    assert g_v == pytest.approx(-0.5 + 2.0 * SIN2_THETA_W, rel=1e-12)
    widths = z_partial_widths()
    total = sum(widths.values())
    hadron = sum(widths[pdg] for pdg in (1, 2, 3, 4, 5))
    invisible = sum(widths[pdg] for pdg in (12, 14, 16))
    # PDG: hadrons 69.91%, invisible 20.00%, one charged lepton 3.363%.
    assert 0.685 < hadron / total < 0.710
    assert 0.195 < invisible / total < 0.210
    assert widths[11] / total == pytest.approx(0.03363, abs=0.0015)
    assert widths[11] == pytest.approx(0.08398, rel=0.02)
    assert total == pytest.approx(GAMMA_Z, rel=0.03)


def test_fermion_pair_matches_breit_wigner_and_low_energy_qed():
    widths = z_partial_widths()
    total = sum(widths.values())
    sigma = fermion_pair_sigma(M_Z, 11, 13, width=total)
    alpha_z = alpha_em(M_Z**2)
    photon = (4.0 * math.pi * alpha_z * alpha_z / (3.0 * M_Z**2)) * (
        fermion_pair_amplitudes(M_Z, 11, 13, width=total)[3]
        * (3.0 - fermion_pair_amplitudes(M_Z, 11, 13, width=total)[3] ** 2)
        / 2.0
    )
    # The Breit–Wigner peak is the resonant piece. The improved Born also
    # contains the non-resonant photon, about 0.5% of the peak.
    breit_wigner = 12.0 * math.pi / M_Z**2 * (widths[11] * widths[13]) / total**2
    assert sigma - photon == pytest.approx(breit_wigner, rel=0.002)

    # Classic result: σ(e+e- → μ+μ-) ≈ (86.8 nb GeV²) / s well below the Z with α(0).
    sqrt_s = 10.0
    qed = (4.0 * math.pi * ALPHA * ALPHA / (3.0 * sqrt_s**2)) * to_pb(1.0) / 1000.0
    assert sqrt_s**2 * qed == pytest.approx(86.8, rel=0.005)
    # The improved Born uses the running α(s): the point cross section scales by (α(s)/α(0))².
    running = qed * (alpha_em(sqrt_s**2) / ALPHA) ** 2
    full_nb = to_pb(fermion_pair_sigma(sqrt_s, 11, 13)) / 1000.0
    assert full_nb == pytest.approx(running, rel=0.005)


def test_forward_backward_asymmetry_at_and_below_the_z():
    _q, g_v, g_a = couplings(13)
    a_f = 2.0 * g_v * g_a / (g_v * g_v + g_a * g_a)
    expected = 0.75 * a_f * a_f
    a_v, a_a, a_1, beta = fermion_pair_amplitudes(M_Z, 11, 13)
    xs = np.linspace(-1.0, 1.0, 4001)
    ys = np.array([fermion_angular_density(float(c), a_v, a_a, a_1, beta) for c in xs])
    forward = _trap(ys[xs >= 0.0], xs[xs >= 0.0])
    backward = _trap(ys[xs <= 0.0], xs[xs <= 0.0])
    asymmetry = (forward - backward) / (forward + backward)
    assert asymmetry == pytest.approx(expected, abs=0.002)
    assert ys.min() >= -1.0e-8

    low = fermion_pair_amplitudes(35.0, 11, 13)
    ys_low = np.array([fermion_angular_density(float(c), *low) for c in xs])
    # PETRA measured a negative muon asymmetry below the pole.
    fwd = _trap(ys_low[xs >= 0.0], xs[xs >= 0.0])
    bak = _trap(ys_low[xs <= 0.0], xs[xs <= 0.0])
    assert (fwd - bak) / (fwd + bak) < 0.0


def test_chi_is_imaginary_on_the_pole_and_real_below_it():
    on_pole = chi_z(M_Z * M_Z)
    assert on_pole.real == pytest.approx(0.0, abs=1e-9)
    assert on_pole.imag < 0.0
    below = chi_z(40.0**2)
    assert below.real < 0.0
    assert abs(below.imag) < abs(below.real)


def test_diphoton_analytic_integral_and_bhabha_ninety_degrees():
    sqrt_s = 40.0
    numeric = integrate_dsigma_domega(sqrt_s, diphoton_dsigma_domega)
    assert diphoton_sigma(sqrt_s) == pytest.approx(numeric, rel=1e-3)

    s = sqrt_s**2
    at_ninety = bhabha_dsigma_domega(sqrt_s, 0.0)
    assert at_ninety == pytest.approx(2.25 * ALPHA * ALPHA / s, rel=1e-6)
    assert bhabha_sigma(sqrt_s) > fermion_pair_sigma(sqrt_s, 11, 13)


def test_higgsstrahlung_benchmark_and_angular_integral():
    fb = to_pb(higgsstrahlung_sigma(250.0)) * 1000.0
    # Unpolarized Born benchmark: 225.59 fb (Phys. Rev. D 100, 073002).
    # This G_F-scheme formula is expected in that neighborhood, not on top of it.
    assert 180.0 < fb < 300.0
    assert higgsstrahlung_sigma(M_Z + M_H) == 0.0
    assert higgsstrahlung_sigma(250.0) > higgsstrahlung_sigma(220.0)
    assert higgsstrahlung_sigma(250.0) > higgsstrahlung_sigma(450.0)

    s = 250.0**2
    lam = kallen(s, M_H**2, M_Z**2) / (s * s)
    r = M_Z**2 / s
    expected_integral = (4.0 / 3.0) * (lam + 12.0 * r)
    xs = np.linspace(-1.0, 1.0, 4001)
    ys = np.array([higgsstrahlung_density(float(c), 250.0) for c in xs])
    assert _trap(ys, xs) == pytest.approx(expected_integral, rel=1e-4)
    # Central production: the weight at 90° exceeds the weight at 0°.
    assert higgsstrahlung_density(0.0, 250.0) > higgsstrahlung_density(1.0, 250.0)


def test_quark_pair_carries_color_and_charge():
    uu = fermion_pair_sigma(40.0, 11, 2)
    mm = fermion_pair_sigma(40.0, 11, 13)
    # Photon-dominated ratio approaches 3*(2/3)^2 * (1+α_s/π).
    qcd = 1.0 + alpha_s(40.0) / math.pi
    assert uu / mm == pytest.approx(3.0 * (4.0 / 9.0) * qcd, rel=0.03)


def test_w_leptonic_branching_fraction_is_near_the_pdg():
    table = branching_table_w()
    electron = next(br for br, a, b in table if a == -11 and b == 12)
    # PDG 2026: B(W → e ν) = 10.71 ± 0.16 %, lepton-universal average 10.86 ± 0.09 %.
    assert electron == pytest.approx(0.1086, abs=0.002)
    assert sum(br for br, _a, _b in table) == pytest.approx(1.0)
    assert sum(br for br, _a, _b in branching_table_z()) == pytest.approx(1.0)


def test_higgs_and_tau_branching_fractions_sum_to_one():
    assert sum(br for _name, br in HIGGS_BR) == pytest.approx(1.0)
    assert sum(br for _name, br in TAU_BR) == pytest.approx(1.0)
    assert max(HIGGS_BR, key=lambda item: item[1])[0] == "bb"


def test_michel_spectrum_mean_and_muon_decay_conservation():
    rng = np.random.default_rng(7)
    xs = np.array([michel_x(rng) for _ in range(20000)])
    assert xs.mean() == pytest.approx(0.70, abs=0.01)
    parent = FourVector(species(13).mass, 0.0, 0.0, 0.0)
    for seed in range(30):
        daughters = leptonic_decay(parent, 11, -12, 14, np.random.default_rng(seed))
        total = daughters[0][1] + daughters[1][1] + daughters[2][1]
        assert total.almost_equal(parent, 1e-8)


def test_tracks_curve_with_the_lorentz_force():
    radius = curvature_radius(1.0, 1.0, 3.8)
    assert radius == pytest.approx(1.0 / (0.299792458 * 3.8), rel=1e-12)
    assert curvature_radius(2.0, 1.0, 3.8) == pytest.approx(2.0 * radius)
    # Fractional quark charge bends less.
    assert curvature_radius(1.0, 2.0 / 3.0, 3.8) == pytest.approx(radius / (2.0 / 3.0))
    positive = track_polyline(1.0, 0.0, +1.0, 3.8, r_stop=5.0, step_m=0.005, max_turns=1.0)
    negative = track_polyline(1.0, 0.0, -1.0, 3.8, r_stop=5.0, step_m=0.005, max_turns=1.0)
    assert positive.points[5][0] > 0.0
    assert positive.points[5][1] < 0.0
    assert negative.points[5][1] > 0.0
    straight = track_polyline(1.0, 0.0, 0.0, 3.8, r_stop=2.0)
    assert straight.radius_m is None
    assert straight.points[-1] == pytest.approx((2.0, 0.0))
    # A 0.3 GeV track in 3.8 T never reaches a 2 m calorimeter.
    soft = track_polyline(0.3, 0.0, 1.0, 3.8, r_stop=2.0, max_turns=2.5)
    assert soft.reached_stop is False
    assert max(math.hypot(x, y) for x, y in soft.points) == pytest.approx(2.0 * curvature_radius(0.3, 1.0, 3.8), rel=0.02)


def _assert_event(event):
    report = __import__("smlab.generator", fromlist=["conservation_report"]).conservation_report(event)
    assert report.ok, report
    assert event.sqrt_s_hat <= event.sqrt_s + 1e-6
    assert len(event.finals()) >= 2


@pytest.mark.parametrize(
    ("process_id", "beam", "sqrt_s", "isr", "force_muon"),
    [
        ("ff13", "ee", 91.188, False, False),
        ("ff13", "ee", 91.188, True, False),
        ("ff13", "ee", 120.0, True, False),
        ("ff15", "ee", 91.188, True, False),
        ("ff12", "ee", 91.188, False, False),
        ("ff5", "ee", 91.188, True, False),
        ("ff6", "ee", 500.0, True, False),
        ("bhabha", "ee", 91.188, False, False),
        ("diphoton", "ee", 40.0, False, False),
        ("zh", "ee", 250.0, True, False),
        ("ff13", "uu", 91.188, False, False),
        ("ff11", "mumu", 91.188, True, False),
        ("ff13", "ee", 2.0, False, True),
    ],
)
def test_generated_events_conserve_everything(process_id, beam, sqrt_s, isr, force_muon):
    process = process_by_id(process_id)
    for seed in range(4):
        event = generate_event(
            process,
            BEAMS[beam],
            sqrt_s,
            seed=1000 + seed,
            isr=isr,
            force_muon=force_muon,
        )
        _assert_event(event)
        again = generate_event(
            process,
            BEAMS[beam],
            sqrt_s,
            seed=1000 + seed,
            isr=isr,
            force_muon=force_muon,
        )
        assert again.particles[3].p4.almost_equal(event.particles[3].p4, 1e-9)


def test_dimuon_at_the_z_has_no_missing_transverse_momentum():
    event = generate_event(process_by_id("ff13"), "ee", M_Z, seed=3, isr=False)
    from smlab.generator import conservation_report

    report = conservation_report(event)
    assert report.missing_px == pytest.approx(0.0, abs=1e-8)
    assert report.missing_py == pytest.approx(0.0, abs=1e-8)
    assert event.sqrt_s_hat == pytest.approx(M_Z)


def test_neutrino_event_has_missing_momentum_and_tau_event_decays():
    neutrinos = generate_event(process_by_id("ff12"), "ee", M_Z, seed=4, isr=False)
    assert all(abs(p.pdg) == 12 for p in neutrinos.finals())
    taus = generate_event(process_by_id("ff15"), "ee", M_Z, seed=5, isr=False)
    assert not any(abs(p.pdg) == 15 for p in taus.finals())
    assert any(p.status == "intermediate" and abs(p.pdg) == 15 for p in taus.particles)


def test_isr_lowers_the_peak_and_feeds_the_radiative_return():
    process = process_by_id("ff13")
    born_peak = sigma_pb(process, "ee", M_Z, isr=False)
    isr_peak = sigma_pb(process, "ee", M_Z, isr=True)
    born_above = sigma_pb(process, "ee", 110.0, isr=False)
    isr_above = sigma_pb(process, "ee", 110.0, isr=True)
    assert isr_peak < born_peak
    assert isr_above > born_above
    # Events generated above the pole should sometimes come back to the Z.
    masses = []
    for seed in range(40):
        event = generate_event(process, "ee", 110.0, seed=seed, isr=True)
        masses.append(event.sqrt_s_hat)
    assert min(masses) < 100.0


def test_tau_decay_from_rest_conserves_charge_and_momentum():
    parent = FourVector(species(15).mass, 0.0, 0.0, 0.0)
    for seed in range(25):
        daughters = decay_tau(parent, 15, np.random.default_rng(seed))
        total = FourVector(0.0, 0.0, 0.0, 0.0)
        charge = 0
        for pdg, p4 in daughters:
            total = total + p4
            charge += species(pdg).charge_thirds
        assert total.almost_equal(parent, 1e-6)
        assert charge == species(15).charge_thirds


def test_g_f_scheme_peak_is_a_few_nanobarns():
    # With the PDG width in the propagator the muon-pair peak is near 2 nb.
    pb = to_pb(fermion_pair_sigma(M_Z, 11, 13))
    assert pb == pytest.approx(2000.0, rel=0.25)


# ---------------------------------------------------------------------------
# Checks added with the 2026 audit
# ---------------------------------------------------------------------------


def test_conversion_constant_is_the_exact_hbar_c_squared():
    # (ħc)² = 0.3893793721 GeV² mb (PDG 2026 Table 1.1).
    assert PB_PER_GEV2 == pytest.approx(0.3893793721e9, rel=1e-12)


def test_running_alpha_reaches_the_on_shell_value_at_the_z():
    assert 1.0 / alpha_em(M_Z**2) == pytest.approx(128.95, abs=0.1)
    assert alpha_em(10.0**2) < alpha_em(M_Z**2) < alpha_em(250.0**2)


def test_alpha_s_is_continuous_across_flavor_thresholds():
    from smlab.constants import M_B, M_C, M_T

    for threshold in (M_C, M_B, M_T):
        below = alpha_s(threshold * (1.0 - 1.0e-9))
        above = alpha_s(threshold * (1.0 + 1.0e-9))
        assert below == pytest.approx(above, rel=1e-6)
    assert alpha_s(10.0) > alpha_s(M_Z) > alpha_s(500.0)


def test_forward_backward_term_scales_as_beta_squared_for_massive_quarks():
    a_v, a_a, a_1, beta = fermion_pair_amplitudes(360.0, 11, 6)
    odd = 0.5 * (fermion_angular_density(1.0, a_v, a_a, a_1, beta) - fermion_angular_density(-1.0, a_v, a_a, a_1, beta))
    assert odd == pytest.approx(a_1 * beta * beta, rel=1e-12)


def test_electron_neutrino_pair_includes_w_exchange():
    s2 = SIN2_THETA_W
    g_l, g_r = -0.5 + s2, s2
    for sqrt_s in (0.5, 1.0):
        s = sqrt_s * sqrt_s
        low_energy = G_F**2 * s / (6.0 * math.pi) * ((g_l + 1.0) ** 2 + g_r**2)
        assert neutrino_w_sigma(sqrt_s) == pytest.approx(low_energy, rel=2e-4)
    # Without the W the Z-only formula is recovered: compare against ν_μ.
    process_e = process_by_id("ff12")
    process_mu = process_by_id("ff14")
    assert process_e.has_w_exchange(BEAMS["ee"])
    assert not process_mu.has_w_exchange(BEAMS["ee"])
    assert process_mu.born_sigma(BEAMS["ee"], 200.0) == pytest.approx(fermion_pair_sigma(200.0, 11, 14))
    # Far above the Z the t-channel W dominates the electron-neutrino rate.
    assert process_e.born_sigma(BEAMS["ee"], 200.0) > 10.0 * process_mu.born_sigma(BEAMS["ee"], 200.0)
    # The W makes the neutrino go forward, along the incoming electron.
    assert neutrino_w_dsigma_domega(200.0, 0.9) > neutrino_w_dsigma_domega(200.0, -0.9)


def test_isr_radiator_is_normalized_and_the_z_peak_matches_lep():
    from smlab.generator import IsrTable
    from smlab.kinematics import isr_exponent, isr_weight

    beta = isr_exponent(M_Z, species(11).mass, ALPHA)
    us = np.linspace(0.0, 1.0, 200001)
    ws = np.array([isr_weight(float(u) ** (1.0 / beta) if u > 0 else 0.0, beta) for u in us])
    assert _trap(ws, us) == pytest.approx(1.0, abs=1e-6)

    process = process_by_id("ff13")
    # LEP pole fit (PDG 2026 Electroweak review): σ⁰_had / R_μ = 41.480 nb / 20.784.
    born_peak = sigma_pb(process, "ee", M_Z, isr=False)
    assert born_peak == pytest.approx(41480.0 / 20.784, rel=0.015)
    # Initial-state radiation lowers the observable peak by roughly a quarter.
    ratio = sigma_pb(process, "ee", M_Z, isr=True) / born_peak
    assert 0.70 < ratio < 0.78
    born = lambda shat: process.born_sigma(BEAMS["ee"], shat)
    coarse = IsrTable(born, 200.0, species(11).mass, 480).total
    fine = IsrTable(born, 200.0, species(11).mass, 6000).total
    assert coarse == pytest.approx(fine, rel=1e-3)


def test_angular_sampling_is_continuous():
    process = process_by_id("ff13")
    rng = np.random.default_rng(11)
    values = {round(process.sample(BEAMS["ee"], M_Z, rng).cos_theta, 12) for _ in range(300)}
    assert len(values) == 300


def test_higgs_to_ww_star_has_one_off_shell_boson():
    from smlab.constants import GAMMA_W, M_W
    from smlab.decays import sample_virtual_pair

    rng = np.random.default_rng(5)
    pairs = [sample_virtual_pair(M_H, M_W, GAMMA_W, rng) for _ in range(3000)]
    assert all(a + b < M_H for a, b in pairs)
    heavy = np.array([max(a, b) for a, b in pairs])
    light = np.array([min(a, b) for a, b in pairs])
    assert np.median(heavy) == pytest.approx(M_W, abs=3.0)
    assert 20.0 < light.mean() < 45.0


def test_three_prong_tau_channel_has_three_charged_pions():
    from smlab.decays import _decay_tau_channel

    parent = FourVector(species(15).mass, 0.0, 0.0, 0.0)
    daughters = _decay_tau_channel(parent, 1, "three_prong_pi0", np.random.default_rng(2))
    charged = [pdg for pdg, _p4 in daughters if abs(pdg) == 211]
    assert len(charged) == 3
    assert sum(species(pdg).charge_thirds for pdg, _p4 in daughters) == -3


def test_neutrino_events_from_electron_beams_conserve_everything():
    for seed in range(6):
        event = generate_event(process_by_id("ff12"), "ee", 200.0, seed=seed, isr=True)
        _assert_event(event)
