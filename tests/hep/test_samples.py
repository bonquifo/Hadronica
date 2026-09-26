"""MadGraph5_aMC@NLO samples: FxFx merging in PYTHIA, and MadSpin top decays read from the LHE file.

References:
    FxFx: R. Frederix, S. Frixione, JHEP 12 (2012) 061.
    MadSpin: P. Artoisenet, R. Frederix, O. Mattelaer, R. Rietkerk, JHEP 03 (2013) 015.
    t t̄ spin correlation D at 13 TeV: −0.24 in the Standard Model (NLO QCD with weak corrections;
        D = −(C(n,n) + C(r,r) + C(k,k))/3 from Table 7 and Eq. 4.23 of W. Bernreuther, D. Heisler,
        Z.-G. Si, JHEP 12 (2015) 026). Uncorrelated decays give D = 0.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from hep_paths import ROOT, sample_path
from hadronica import constants as C


def _info(name: str) -> dict:
    with open(os.path.join(os.path.dirname(sample_path(name)), "info.json"), encoding="utf-8") as handle:
        return json.load(handle)


# -- FxFx ---------------------------------------------------------------------------

def _fxfx_run(n: int, sqrt_s: float = 13000.0):
    from worker import Worker

    w = Worker()
    w.init({"beams": "pp", "process": "pp_z_ll", "sqrt_s": sqrt_s, "seed": 17, "decays": "generator",
            "source": "nlo", "sample": "dy_fxfx"})
    reply = w.generate(n, jets=False)
    info = w._info()
    return reply, info.nTried(), info.nAccepted()


def test_without_the_matching_hook_nothing_is_merged(monkeypatch):
    # The control: PYTHIA's JetMatching settings alone do nothing from Python.
    sample_path("dy_fxfx_13000")
    import hadronica_fxfx

    monkeypatch.setattr(hadronica_fxfx, "attach", lambda pythia: None)
    reply, tried, accepted = _fxfx_run(300)
    assert tried == accepted
    lhe_sigma = _info("dy_fxfx_13000")["sigma_pb"]
    assert reply["sigma_pb"] == pytest.approx(lhe_sigma, rel=0.25)  # unmerged: MadGraph's σ before the veto


def test_fxfx_veto_merges_the_jet_multiplicities():
    sample_path("dy_fxfx_13000")
    reply, tried, accepted = _fxfx_run(2000)
    acceptance = accepted / tried
    # Binomial error on ~4500 tries is 0.0074; the two production samples gave 0.444 and 0.443.
    assert 0.40 < acceptance < 0.49
    merged, error = reply["sigma_pb"], reply["sigma_err_pb"]
    # Merging removes the double counting: well below MadGraph's σ before the veto (7.2 nb) ...
    assert merged < 0.7 * _info("dy_fxfx_13000")["sigma_pb"]
    # ... and lands within merging-scale and higher-order uncertainties (±20 %) of the
    # independent inclusive NLO calculation for the same cuts (m_ℓℓ > 60 GeV, e + μ).
    inclusive = _info("dy_13000")["sigma_pb"]
    assert 0.8 < merged / inclusive < 1.2
    # And consistent with the 60 000-event validation run.
    results = json.load(open(os.path.join(ROOT, "hep", "validation", "results.json"), encoding="utf-8"))
    stored = results["benchmarks"]["lhc_z_pt_fxfx"]["sigma_pb"]
    assert abs(merged - stored) < 4.0 * error


def test_merged_cross_section_grows_with_energy_like_the_inclusive_nlo():
    sample_path("dy_fxfx_13600")
    sample_path("dy_13600")
    reply, _tried, _accepted = _fxfx_run(5000, 13600.0)
    results = json.load(open(os.path.join(ROOT, "hep", "validation", "results.json"), encoding="utf-8"))
    at_13 = results["benchmarks"]["lhc_z_pt_fxfx"]["sigma_pb"]
    merged_ratio = reply["sigma_pb"] / at_13
    inclusive_ratio = _info("dy_13600")["sigma_pb"] / _info("dy_13000")["sigma_pb"]
    # Statistical error on the ratio ≈ 1.3 %; allow ~4σ.
    assert merged_ratio == pytest.approx(inclusive_ratio, abs=0.05)


# -- MadSpin (read directly from the LHE file) ------------------------------------------

def _lhe_events(path: str):
    """Yield (weight, particles) with particles = [(id, status, mother1, px, py, pz, E, m)], mothers 1-based."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = iter(handle)
        for line in lines:
            if not line.startswith("<event"):
                continue
            header = next(lines).split()
            count, weight = int(header[0]), float(header[2])
            particles = []
            for _ in range(count):
                f = next(lines).split()
                particles.append((int(f[0]), int(f[1]), int(f[2]), float(f[6]), float(f[7]), float(f[8]),
                                  float(f[9]), float(f[10])))
            yield weight, particles


def _mass(p) -> float:
    e, px, py, pz = p
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def _add(*vectors):
    return tuple(sum(v[k] for v in vectors) for k in range(4))


def _four(particle):
    return (particle[6], particle[3], particle[4], particle[5])


def _boost(p, frame):
    """p in the rest frame of the four-vector ``frame``."""
    e, px, py, pz = p
    fe, fx, fy, fz = frame
    m = _mass(frame)
    bx, by, bz = fx / fe, fy / fe, fz / fe
    b2 = bx * bx + by * by + bz * bz
    if b2 <= 0.0:
        return p
    gamma = fe / m
    bp = bx * px + by * py + bz * pz
    k = (gamma - 1.0) * bp / b2 - gamma * e
    return (gamma * (e - bp), px + k * bx, py + k * by, pz + k * bz)


@pytest.fixture(scope="module")
def madspin():
    path = sample_path("ttbar_ms_13000")
    return list(_lhe_events(path))


LEPTONS = {11, 13, 15}


def _decays(particles):
    """{top index: (W, b, charged lepton or None, W daughters, top daughters)} for both tops (indices)."""
    out = {}
    for i, p in enumerate(particles):
        if abs(p[0]) == 6 and p[1] == 2:
            kids = [j for j, q in enumerate(particles) if q[2] == i + 1]
            w = next(j for j in kids if abs(particles[j][0]) == 24)
            b = next(j for j in kids if abs(particles[j][0]) == 5)
            w_kids = [j for j, q in enumerate(particles) if q[2] == w + 1]
            lepton = next((j for j in w_kids if abs(particles[j][0]) in LEPTONS), None)
            out[i] = (w, b, lepton, w_kids, kids)
    return out


def test_madspin_keeps_the_nlo_cross_section_and_every_event(madspin):
    decayed, undecayed = _info("ttbar_ms_13000"), _info("ttbar_13000")
    assert decayed["madspin"] is True
    # Off-shell MadSpin biased the top line shape (+0.43 GeV); production samples are on-shell.
    assert decayed.get("spinmode") == "onshell"
    assert decayed["events"] == undecayed["events"] == len(madspin)
    # All decay channels are kept (branching ratios sum to one), so σ is unchanged.
    assert decayed["sigma_from_events_pb"] == pytest.approx(undecayed["sigma_from_events_pb"], rel=1e-4)


def test_every_decayed_event_conserves_four_momentum(madspin):
    worst = 0.0
    for _weight, particles in madspin:
        incoming = _add(*[_four(p) for p in particles if p[1] == -1])
        outgoing = _add(*[_four(p) for p in particles if p[1] == 1])
        worst = max(worst, max(abs(a - b) for a, b in zip(incoming, outgoing)) / incoming[0])
    assert worst < 1e-6


# MadGraph derives M_W in the G_F scheme from M_Z, G_F and α⁻¹ = 132.04: 80.3617 GeV.
M_W_MG5 = 80.3617
# MadSpin's own leading-order widths (printed in the decayed sample's banner).
GAMMA_T_LO, GAMMA_W_LO = 1.4839, 2.0436


def test_top_and_w_decays_have_the_input_masses(madspin):
    """On-shell tops sit exactly at m_t; off-shell lines are symmetric Breit–Wigners on the input masses.

    The off-shell expectation is MadGraph's full LO t t̄ → bW bW matrix element with the same
    inputs (median 172.62 GeV, 3589 vs 3606 tops within one width below and above the pole).
    """
    spinmode = _info("ttbar_ms_13000").get("spinmode", "madspin")
    tops, ws = [], []
    radiative = 0
    for _weight, particles in madspin:
        decays = _decays(particles)
        assert len(decays) == 2
        for top, (w, _b, _lepton, w_kids, top_kids) in decays.items():
            m_top = _mass(_four(particles[top]))
            m_w = _mass(_four(particles[w]))
            # Each decay conserves four-momentum: t → b W, or radiatively t → b W γ.
            assert _mass(_add(*[_four(particles[j]) for j in top_kids])) == pytest.approx(m_top, abs=1e-3)
            if len(top_kids) == 3:
                assert {abs(particles[j][0]) for j in top_kids} == {5, 24, 22}
                radiative += 1
            if not w_kids:  # MadSpin left this W for PYTHIA to decay
                assert particles[w][1] == 1
                continue
            assert len(w_kids) == 2
            assert _mass(_add(*[_four(particles[j]) for j in w_kids])) == pytest.approx(m_w, abs=1e-3)
            tops.append(m_top)
            ws.append(m_w)
    # Radiative t → b W γ decays: about 1 % (1361 of 120 000 tops), their W decayed later by PYTHIA.
    assert radiative < 0.02 * 2 * len(madspin)
    shapes = [(ws, M_W_MG5, GAMMA_W_LO)]
    if spinmode == "onshell":
        # Tops exactly on shell; their W decays (generated by MadEvent) keep the full Breit–Wigner.
        assert max(abs(m - C.M_T) for m in tops) < 1e-3
    else:
        shapes.append((tops, C.M_T, GAMMA_T_LO))
    for masses, pole, width in shapes:
        below = sum(pole - width <= m < pole for m in masses)
        above = sum(pole <= m < pole + width for m in masses)
        assert abs(above - below) < 4.0 * math.sqrt(above + below), (pole, below, above)
        assert sorted(masses)[len(masses) // 2] == pytest.approx(pole, abs=0.1)


def test_w_decays_to_each_lepton_flavour_one_time_in_nine(madspin):
    counts = {11: 0, 13: 0, 15: 0}
    n_w = 0
    for _weight, particles in madspin:
        for _top, (_w, _b, lepton, w_kids, _top_kids) in _decays(particles).items():
            if not w_kids:
                continue  # left for PYTHIA to decay
            n_w += 1
            if lepton is not None:
                counts[abs(particles[lepton][0])] += 1
    # Tree level, massless: 3 lepton + 2 × 3 quark channels → 1/9 each (the τ-mass correction is 0.07 %).
    sigma = math.sqrt(n_w * (1 / 9) * (8 / 9))
    for flavour, count in counts.items():
        assert abs(count - n_w / 9) < 4 * sigma, (flavour, count, n_w / 9)


def test_top_spin_correlation_matches_the_standard_model(madspin):
    """D = −3 ⟨cos φ⟩, φ between ℓ⁺ in the t rest frame and ℓ⁻ in the t̄ rest frame (via the t t̄ frame)."""
    sum_w = sum_wc = sum_w2c2 = 0.0
    samples = []
    for weight, particles in madspin:
        decays = _decays(particles)
        if any(lepton is None for _w, _b, lepton, _k, _t in decays.values()):
            continue
        tops = {particles[i][0]: i for i in decays}
        t, tbar = _four(particles[tops[6]]), _four(particles[tops[-6]])
        system = _add(t, tbar)
        directions = []
        for top_id, top_p in ((6, t), (-6, tbar)):
            lepton = particles[decays[tops[top_id]][2]]
            in_pair = _boost(_four(lepton), system)
            in_top = _boost(in_pair, _boost(top_p, system))
            norm = math.sqrt(sum(c * c for c in in_top[1:]))
            directions.append([c / norm for c in in_top[1:]])
        cos_phi = sum(a * b for a, b in zip(*directions))
        samples.append((weight, cos_phi))
        sum_w += weight
        sum_wc += weight * cos_phi
    mean = sum_wc / sum_w
    # Signed MC@NLO weights: error from the weighted variance.
    sum_w2c2 = sum(w * w * (c - mean) ** 2 for w, c in samples)
    error = math.sqrt(sum_w2c2) / abs(sum_w)
    d, d_error = -3.0 * mean, 3.0 * error
    assert len(samples) > 5000  # dileptonic, e/μ/τ: (1/3)² of 60 000
    assert d < -4.0 * d_error  # spin correlation present (uncorrelated decays give D = 0)
    assert -0.34 < d < -0.14  # Standard Model: D = −0.243 (Bernreuther, Heisler, Si)
