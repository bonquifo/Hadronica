"""Shower re-tune: fit-region χ² and the quadratic χ² surface (hep/tune.py)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hep"))

import tune  # noqa: E402


def _plot(path, edges, data, mc):
    return {"path": path, "edges": edges, "data": data, "data_err": [0.1] * len(data),
            "mc": mc, "mc_err": [0.0] * len(mc)}


def test_region_chi2_uses_only_the_fit_region_and_the_shape():
    edges = [0.0, 10.0, 20.0, 30.0, 40.0]
    data = [1.0, 2.0, 1.0, 5.0]
    # Twice the data everywhere below 30 GeV (normalization only) and wrong above it.
    mc = [2.0, 4.0, 2.0, 0.0]
    chi2, bins = tune.region_chi2([_plot("/ATLAS_2019_I1768911/d27-x01-y01", edges, data, mc),
                                   _plot("/ATLAS_2019_I1768911/d99-x01-y01", edges, data, mc)])
    assert bins == 3
    assert chi2 == pytest.approx(0.0, abs=1e-12)
    shifted = [1.0, 2.2, 1.0, 5.0]
    chi2, _ = tune.region_chi2([_plot("/ATLAS_2019_I1768911/d27-x01-y01", edges, data, shifted)])
    assert chi2 > 1.0


def test_quadratic_fit_finds_the_minimum_and_its_width():
    points = [(x, y, 5.0 + (x - 1.6) ** 2 / 0.2 ** 2 + (y - 1.1) ** 2 / 0.5 ** 2)
              for x in tune.GRID[tune.KT] for y in tune.GRID[tune.PT0]]
    fit = tune.fit_quadratic(points)
    assert fit["minimum"] == pytest.approx([1.6, 1.1], abs=1e-6)
    assert fit["chi2_min"] == pytest.approx(5.0, abs=1e-6)
    assert fit["sigma"] == pytest.approx([0.2, 0.5], rel=1e-6)


def test_quadratic_fit_reports_a_saddle_as_no_minimum():
    points = [(x, y, x * x - y * y) for x in tune.GRID[tune.KT] for y in tune.GRID[tune.PT0]]
    assert tune.fit_quadratic(points)["minimum"] is None


def test_app_requests_the_tune_only_for_samples_it_applies_to():
    import pygame

    from hadronica.app import LabApp

    application = LabApp(size=(1480, 900), headless=True, seed=7)
    try:
        application.engine_kind = "pythia"
        application.py_beam = "pp"
        application.py_process = "pp_z_ll"
        application.py_energy["pp"] = 13000.0
        application.py_nlo_samples = {"pp_z_ll@13000": {"sample": "dy_fxfx", "sigma_pb": 1900.0},
                                      "pp_ttbar@13000": {"sample": "ttbar_ms", "sigma_pb": 663.0}}
        assert "tune" not in application._py_config()  # no tune.json yet
        application.py_tune = {"name": "Hadronica-AZ 2026", "settings": {tune.KT: 1.5, tune.PT0: 1.2},
                                "applies_to": ["dy_fxfx", "dy"]}
        config = application._py_config()
        assert config["source"] == "nlo" and config["tune"] == "hadronica"
        hash(application._py_key(config))
        application.py_process = "pp_ttbar"
        assert "tune" not in application._py_config()
        application.py_process = "pp_z_ll"
        application.py_options["tune"] = False
        assert "tune" not in application._py_config()
        application.py_options["tune"] = True
        application.py_options["nlo"] = False
        assert "tune" not in application._py_config()
    finally:
        pygame.quit()


def test_profile_fit_when_the_minimum_sits_on_a_parameter_limit():
    # χ² falls towards the lowest p_T0 (a hard limit) and has a clear minimum in k_T there.
    rows = [{"settings": {tune.KT: kt, tune.PT0: pt0}, "chi2": 40.0 + 30.0 * (kt - 3.1) ** 2 + 40.0 * (pt0 - 0.5)}
            for kt in (2.2, 2.6, 3.0, 3.4, 3.8) for pt0 in (0.5, 0.75, 1.0)]
    settings, method, _fit, profile = tune.choose(rows)
    assert settings[tune.PT0] == 0.5
    assert settings[tune.KT] == pytest.approx(3.1, abs=0.01)
    assert profile["fixed"] == {tune.PT0: 0.5} and "fixed" in method
