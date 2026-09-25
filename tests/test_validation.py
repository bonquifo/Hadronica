"""Validation panel, signed-weight histograms, and pileup display."""

from __future__ import annotations

import math

import pygame
import pytest

from smlab.app_pythia import _auto_hist
from smlab.validation_view import grade_color, grouped, mc_limited
from smlab.theme import BAD, GOOD, WARN


def _plot(path, chi2, n=4):
    edges = [float(i) for i in range(n + 1)]
    data = [10.0, 8.0, 5.0, 2.0][:n]
    return {
        "path": path, "title": "Charged multiplicity", "xlabel": "$N_\\mathrm{ch}$", "ylabel": "",
        "chi2_ndf": chi2, "shape_chi2_ndf": chi2 / 2, "norm_ratio": 1.02, "ndf": n,
        "edges": edges, "data": data, "data_err": [0.5] * n,
        "mc": [v * 1.05 for v in data], "mc_err": [0.2] * n,
    }


def _results():
    lo = {
        "key": "lhc_z_pt", "benchmark": "lhc_z_pt", "variant": "lo", "generator": "PYTHIA 8.3 LO+PS",
        "title": "LHC 13 TeV, Z boson transverse momentum", "tests": "ISR", "analyses": ["ATLAS_2019_I1768911"],
        "reference": "ATLAS, Eur. Phys. J. C 80 (2020) 616", "median_chi2_ndf": 4.2,
        "median_shape_chi2_ndf": 4.2, "median_norm_ratio": 1.0, "plots": [_plot("/A/d01", 4.2)],
    }
    nlo = dict(lo, key="lhc_z_pt_nlo", variant="nlo", generator="MadGraph5_aMC@NLO + PYTHIA 8.3", median_chi2_ndf=1.3,
               plots=[_plot("/A/d01", 1.3)])
    fxfx = dict(lo, key="lhc_z_pt_fxfx", variant="fxfx", generator="FxFx", median_chi2_ndf=0.9,
                plots=[_plot("/A/d01", 0.9)])
    lep = dict(lo, key="lep_z_hadrons", benchmark="lep_z_hadrons", title="LEP", median_chi2_ndf=None, plots=[])
    return {"generated": "2026-09-23 12:00",
            "benchmarks": {"lep_z_hadrons": lep, "lhc_z_pt_fxfx": fxfx, "lhc_z_pt": lo, "lhc_z_pt_nlo": nlo}}


def test_results_are_grouped_by_benchmark_in_variant_order():
    rows = grouped(_results())
    assert [row[0] for row in rows] == ["lep_z_hadrons", "lhc_z_pt"]
    base, entries = rows[1]
    assert [entry["variant"] for entry in entries] == ["lo", "nlo", "fxfx"]
    assert len(rows[0][1]) == 1


def test_chi2_grades():
    assert grade_color(1.2) == GOOD
    assert grade_color(3.0) == WARN
    assert grade_color(40.0) == BAD
    assert grade_color(None) != GOOD


def test_signed_weights_subtract_in_histograms():
    hist = _auto_hist([(1.0, 0.5), (1.0, -0.5), (2.0, 0.5), (3.0, 0.5)], 3)
    assert hist.total == pytest.approx(2.0)
    lo_only = _auto_hist([4.0, 5.0, 6.0], 3)
    assert lo_only.total == pytest.approx(3.0)


def test_validation_panel_draws_lo_and_nlo(monkeypatch):
    from smlab import app as app_module
    from smlab.app import LabApp

    application = LabApp(size=(1480, 900), headless=True, seed=7)
    try:
        monkeypatch.setattr(app_module, "load_results", lambda: (_results(), "test"))
        application._open_validation()
        application.validation_selected = "lhc_z_pt"
        application.draw()
        actions = {action for _rect, action, _payload in application.hot}
        assert {"val-select", "val-plot", "val-run", "close-validation"} <= actions
        next_button = next(rect for rect, action, payload in application.hot if action == "val-plot" and payload == 1)
        application._click(next_button.center)
        assert application.validation_plot == 1
        application.draw()
        close = next(rect for rect, action, _payload in application.hot if action == "close-validation")
        application._click(close.center)
        assert not application.show_validation
    finally:
        pygame.quit()


def test_pileup_tracks_come_from_their_own_vertices():
    from smlab.fullscene import build_traces
    from tests.test_engine import _parse

    event = _parse()
    event.reco = {
        "tracks_z": [
            [5.0, 0.0, 0.3, 1.0, 42.0, 1],   # pileup, from z = 42 mm
            [5.0, 0.5, 1.3, -1.0, 0.0, 0],   # hard-scatter track: already drawn from the truth record
            [0.3, 0.0, 2.0, 1.0, -10.0, 1],  # pileup below the 0.5 GeV display threshold
        ]
    }
    pileup = [trace for trace in build_traces(event, 3.8) if trace.kind == "pileup"]
    assert len(pileup) == 1
    x0, y0, z0 = pileup[0].points[0]
    assert (x0, y0) == (0.0, 0.0) and z0 == pytest.approx(0.042)
    end = pileup[0].points[-1]
    assert math.hypot(end[0], end[1]) == pytest.approx(1.20, abs=0.02)


def test_plots_short_of_simulated_events_are_flagged():
    good = _plot("/A/d01", 1.0)
    assert not mc_limited(good)
    empty_bin = dict(good, mc=[10.0, 0.0, 5.0, 2.0], mc_err=[0.2, 0.0, 0.2, 0.2])
    assert mc_limited(empty_bin)
    # The χ² denominator is σ_data² + σ_MC²: once the MC error dominates, χ² measures MC noise.
    noisy = dict(good, mc_err=[0.6, 0.7, 0.6, 0.1])  # data errors are 0.5
    assert mc_limited(noisy)
    assert not mc_limited(dict(good, mc_err=[0.4, 0.4, 0.6, 0.1]))


def test_a_partial_rerun_updates_its_entries_and_keeps_the_rest(tmp_path, monkeypatch):
    import json

    from smlab import validation_view as view

    bundled = {"generated": "2026-09-24 00:47", "benchmarks": {
        "lep_z_hadrons": {"benchmark": "lep_z_hadrons", "reference": "ALEPH, Phys. Rept. 294 (1998) 1", "median_chi2_ndf": 2.5},
        "lhc_jets": {"benchmark": "lhc_jets", "reference": "CMS, Eur. Phys. J. C 76 (2016) 451", "median_chi2_ndf": 5.7}}}
    rerun = {"generated": "2026-09-25 09:00", "benchmarks": {
        "lep_z_hadrons": {"benchmark": "lep_z_hadrons", "reference": "an outdated citation", "median_chi2_ndf": 2.1}}}
    bundled_path, user_dir = tmp_path / "bundled.json", tmp_path / "user"
    user_dir.mkdir()
    bundled_path.write_text(json.dumps(bundled), encoding="utf-8")
    (user_dir / "results.json").write_text(json.dumps(rerun), encoding="utf-8")
    monkeypatch.setattr(view, "_bundled_results", lambda: str(bundled_path))
    monkeypatch.setattr(view, "user_results_dir", lambda: str(user_dir))
    data, source = view.load_results()
    assert set(data["benchmarks"]) == {"lep_z_hadrons", "lhc_jets"}  # nothing hidden
    assert data["benchmarks"]["lep_z_hadrons"]["median_chi2_ndf"] == 2.1  # the rerun wins
    assert data["benchmarks"]["lep_z_hadrons"]["reference"] == "ALEPH, Phys. Rept. 294 (1998) 1"  # verified citation
    assert data["generated"] == "2026-09-25 09:00" and "bundled" in source


def test_every_stored_result_carries_the_verified_citation():
    import json
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "hep"))
    import validate

    want = {b["key"]: b["reference"] for b in validate.BENCHMARKS}
    stored = json.load(open(os.path.join(root, "hep", "validation", "results.json"), encoding="utf-8"))["benchmarks"]
    for key, entry in stored.items():
        assert entry["reference"] == want[entry["benchmark"]], key
