"""PYTHIA lepton-collider cross sections against measurements and standard predictions.

The references are fixed before the comparison; the tolerances combine the
reference's precision with what a leading-order generator with QED initial-state
radiation can be expected to reach (a few per cent). Statistical errors of the
estimates below are about 1 %.
"""

from __future__ import annotations

import pytest

from worker import Worker

CASES = [
    # (process, √s [GeV], reference [pb], relative tolerance, source)
    ("ll_gmz_had", 91.1876, 30.5e3, 0.04,
     "Z-peak hadronic cross section with initial-state radiation ≈ 30.5 nb: the pole value σ⁰_had = 41.480 nb "
     "(PDG 2026 Electroweak review) lowered about 25 % by ISR, as the LEP line-shape analyses quote"),
    ("ll_ww", 200.0, 17.0, 0.06,
     "σ(e⁺e⁻ → W⁺W⁻) at 200 GeV ≈ 17.0 pb predicted by RacoonWW/YFSWW; LEP measured 16.77 ± 0.29 pb at 199.5 GeV (Phys. Rept. 532 (2013) 119, Tables E.2, E.4)"),
    ("ll_zz", 200.0, 0.99, 0.08,
     "σ(e⁺e⁻ → ZZ, NC02) at 200 GeV ≈ 0.99 pb predicted by YFSZZ/ZZTO; LEP measured 0.95 ± 0.12 pb at 199.5 GeV (Phys. Rept. 532 (2013) 119, Tables E.12, E.13)"),
    ("ll_zh", 240.0, 0.1962, 0.08,
     "σ(e⁺e⁻ → ZH) at 240 GeV = 196.2 fb (CEPC Conceptual Design Report Vol. II, Table 11.2, arXiv:1811.10545)"),
]


@pytest.mark.parametrize("process,sqrt_s,reference_pb,tolerance,source", CASES, ids=[c[0] for c in CASES])
def test_cross_section_matches_the_reference(process, sqrt_s, reference_pb, tolerance, source):
    w = Worker()
    w.init({"beams": "ee", "process": process, "sqrt_s": sqrt_s, "seed": 5})
    reply = w.generate(2500, jets=False)
    sigma, error = reply["sigma_pb"], reply["sigma_err_pb"]
    assert error < 0.02 * sigma  # the estimate itself is precise enough to test
    assert sigma == pytest.approx(reference_pb, rel=tolerance), f"{sigma:.4g} ± {error:.2g} pb vs {source}"
