"""The PYTHIA worker: PDG inputs, tune wiring, sample choice, and the HepMC3 hand-off to Rivet."""

from __future__ import annotations

import json
import os

import pytest

import worker
from smlab import constants as C
from worker import Worker, nlo_samples, smlab_tune

TUNE_PATH = os.path.join(os.path.dirname(worker.__file__), "validation", "tune.json")


def _pp_z(**extra) -> Worker:
    w = Worker()
    w.init({"beams": "pp", "process": "pp_z_ll", "sqrt_s": 13000.0, "seed": 3, "decays": "generator", **extra})
    return w


def test_pythia_runs_with_the_pdg_2026_masses_and_widths():
    data = _pp_z().pythia.particleData
    assert data.m0(23) == pytest.approx(C.M_Z)
    assert data.m0(24) == pytest.approx(C.M_W) and data.mWidth(24) == pytest.approx(C.GAMMA_W)
    assert data.m0(25) == pytest.approx(C.M_H) and data.mWidth(25) == pytest.approx(C.GAMMA_H)
    assert data.m0(6) == pytest.approx(C.M_T) and data.mWidth(6) == pytest.approx(C.GAMMA_T)
    assert data.m0(15) == pytest.approx(C.M_TAU)
    # The width the Breit–Wigner actually uses, at the pole (not just the stored number).
    assert data.particleDataEntryPtr(24).resWidth(24, C.M_W) == pytest.approx(C.GAMMA_W)
    assert data.particleDataEntryPtr(6).resWidth(6, C.M_T) == pytest.approx(C.GAMMA_T)
    # PYTHIA computes the γ*/Z width from its couplings and does not honour a forced value;
    # it must still agree with the PDG average to well under 1 %.
    assert data.mWidth(23) == pytest.approx(C.GAMMA_Z, rel=0.005)


def test_z_pole_hadronic_cross_section_matches_lep():
    # LEP pole fit: σ⁰_had = 41.480 nb (PDG 2026 Electroweak review), here without initial-state radiation.
    w = Worker()
    w.init({"beams": "ee", "process": "ll_gmz_had", "sqrt_s": C.M_Z, "seed": 3, "isr": False})
    sigma_nb = w.generate(200, jets=False)["sigma_pb"] / 1e3
    assert sigma_nb == pytest.approx(41.480, abs=0.1)


def _conserves(particles: list[list]) -> bool:
    beams = [row for row in particles[1:3]]
    finals = [row for row in particles if row[1] > 0]
    # PYTHIA's own tolerance: it warns when the summed deviation exceeds Check:epTolWarn
    # (1e-6) of the collision energy. The worker also rounds each component to 1e-7 GeV.
    bound = 1e-6 * (beams[0][9] + beams[1][9]) + 0.5e-7 * (len(beams) + len(finals))
    for k in (6, 7, 8, 9):  # px, py, pz, E
        total_in = sum(row[k] for row in beams)
        total_out = sum(row[k] for row in finals)
        if abs(total_in - total_out) > bound:
            return False
    return True


def test_generated_lo_events_conserve_four_momentum():
    reply = _pp_z().generate(20, jets=False)
    assert len(reply["events"]) == 20
    assert all(_conserves(event["particles"]) for event in reply["events"])


def test_the_smlab_tune_is_applied_last_and_only_at_nlo():
    tune = json.load(open(TUNE_PATH, encoding="utf-8"))
    expected = [f"{key} = {value}" for key, value in tune["settings"].items()]
    lo = Worker().init({"beams": "pp", "process": "pp_z_ll", "sqrt_s": 13000.0, "seed": 1, "tune": "smlab"})
    assert not any(line in lo["settings"] for line in expected)  # Monash at leading order
    if "pp_z_ll@13000" not in nlo_samples():
        pytest.skip("no NLO Z sample")
    nlo = Worker().init({"beams": "pp", "process": "pp_z_ll", "sqrt_s": 13000.0, "seed": 1, "source": "nlo",
                         "tune": "smlab"})["settings"]
    # After every other setting (only the random seed follows), so nothing overrides it.
    assert nlo[-2 - len(expected):-2] == expected
    assert nlo[-2].startswith("Random:setSeed")


def test_an_explicit_tune_dictionary_overrides_pythia_defaults():
    w = _pp_z(tune={"BeamRemnants:primordialKThard": 2.5})
    assert w.pythia.settings.parm("BeamRemnants:primordialKThard") == pytest.approx(2.5)


def test_hello_reports_the_tune_the_app_will_offer():
    tune = smlab_tune()
    stored = json.load(open(TUNE_PATH, encoding="utf-8"))
    assert tune["settings"] == stored["settings"] and tune["name"] == stored["name"]


def test_best_sample_is_the_merged_one_and_hidden_without_the_hook(monkeypatch):
    samples = nlo_samples(all_variants=True)
    if "pp_z_ll@13000#dy_fxfx" not in samples:
        pytest.skip("no FxFx sample")
    assert nlo_samples()["pp_z_ll@13000"]["sample"] == "dy_fxfx"
    monkeypatch.setattr(worker, "fxfx_available", lambda: False)
    assert "pp_z_ll@13000#dy_fxfx" not in nlo_samples(all_variants=True)
    assert nlo_samples()["pp_z_ll@13000"]["sample"] == "dy"  # showering FxFx unmerged would double count


def test_hepmc_round_trip_keeps_momenta_and_negative_weights(tmp_path):
    # Round 3 regression: pyHepMC3 silently dropped the weights, and MC@NLO needs their sign.
    import pyhepmc as hep

    from detector import to_hepmc

    raw = _pp_z().generate(3, jets=False)["events"]
    path = str(tmp_path / "events.hepmc")
    run_info = hep.GenRunInfo()
    run_info.weight_names = ["Default"]
    weights = [-0.75, 1.25, 1.0]
    with hep.io.WriterAscii(path) as writer:
        for number, (event, weight) in enumerate(zip(raw, weights)):
            record = to_hepmc(event["particles"], number)
            record.run_info = run_info
            record.weights = [weight]
            writer.write(record)
    with hep.io.ReaderAscii(path) as reader:
        read = [event for event in reader]
    assert [list(event.weights) for event in read] == [[w] for w in weights]
    for event, source in zip(read, raw):
        beams = [p for p in event.particles if p.status == 4]
        finals = [p for p in event.particles if p.status == 1]
        assert len(beams) == 2 and len(finals) == sum(1 for row in source["particles"] if row[1] > 0)
        for attr in ("px", "py", "pz", "e"):
            total_in = sum(getattr(p.momentum, attr) for p in beams)
            total_out = sum(getattr(p.momentum, attr) for p in finals)
            assert total_out == pytest.approx(total_in, abs=1e-3)
        # Every non-beam particle has a production vertex: Rivet walks the graph from the beams.
        assert all(p.production_vertex is not None for p in event.particles if p.status != 4)
