"""PYTHIA bridge: parsing, conservation, displaced vertices, and the PYTHIA panels.

The unit tests use a hand-built event in the worker's wire format. The last
test drives the real WSL worker and is skipped when it is not installed.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pygame
import pytest

from hadronica import engine
from hadronica.engine import PythiaEngine, conservation, event_from_reply, worker_path_in_wsl
from hadronica.fullscene import build_traces, count_summary
from hadronica.lorentz import FourVector
from hadronica.report import english_name, proper_lifetime_s, register_species, symbol

M_MU = 0.1056583755
M_PI = 0.13957039
M_KS = 0.497611


def _row(pdg, status, m1, m2, d1, d2, p4, mass, vertex=(0.0, 0.0, 0.0), charge=0.0):
    return [pdg, status, m1, m2, d1, d2, p4.px, p4.py, p4.pz, p4.e, mass, *vertex, 0.0, charge]


def _on_shell(px, py, pz, mass):
    return FourVector(math.sqrt(px * px + py * py + pz * pz + mass * mass), px, py, pz)


def synthetic_reply() -> dict:
    """e+e- → Z → μ+μ-, plus a K0_S that decays to π+π- 10 mm from the vertex."""
    beam_e = 45.6
    e_minus = FourVector(beam_e, 0.0, 0.0, math.sqrt(beam_e**2 - 0.000511**2))
    e_plus = FourVector(beam_e, 0.0, 0.0, -math.sqrt(beam_e**2 - 0.000511**2))
    total = e_minus + e_plus
    pi_plus = _on_shell(1.0, 0.12, 0.0, M_PI)
    pi_minus = _on_shell(1.0, -0.12, 0.0, M_PI)
    kaon = pi_plus + pi_minus
    mu_minus = _on_shell(-20.0, 30.0, 10.0, M_MU)
    mu_plus = total - kaon - mu_minus
    z = total
    decay_at = (10.0, 0.0, 0.0)
    rows = [
        _row(90, -11, 0, 0, 0, 0, total, total.mass),
        _row(11, -12, 0, 0, 3, 3, e_minus, 0.000511, charge=-1.0),
        _row(-11, -12, 0, 0, 3, 3, e_plus, 0.000511, charge=1.0),
        _row(23, -22, 1, 2, 4, 6, z, z.mass),
        _row(13, 1, 3, 3, 0, 0, mu_minus, M_MU, charge=-1.0),
        _row(-13, 1, 3, 3, 0, 0, mu_plus, mu_plus.mass, charge=1.0),
        _row(310, -91, 3, 3, 7, 8, kaon, kaon.mass),
        _row(211, 91, 6, 6, 0, 0, pi_plus, M_PI, decay_at, 1.0),
        _row(-211, 91, 6, 6, 0, 0, pi_minus, M_PI, decay_at, -1.0),
    ]
    hard = [
        [11, -21, 0, 0, 0.0, 0.0, e_minus.pz, e_minus.e, 0.000511],
        [-11, -21, 0, 0, 0.0, 0.0, e_plus.pz, e_plus.e, 0.000511],
        [23, -22, 1, 2, 0.0, 0.0, 0.0, z.e, z.mass],
        [13, 23, 3, 3, mu_minus.px, mu_minus.py, mu_minus.pz, mu_minus.e, M_MU],
        [-13, 23, 3, 3, mu_plus.px, mu_plus.py, mu_plus.pz, mu_plus.e, M_MU],
    ]
    event = {
        "species": {"310": ["K_S0", M_KS, 26.84], "211": ["pi+", M_PI, 7804.5], "-211": ["pi-", M_PI, 7804.5]},
        "particles": rows,
        "hard": hard,
        "jets": [],
        "code": 221,
        "name": "f fbar -> gamma*/Z0",
        "sqrt_s_hat": total.mass,
        "weight": 1.0,
    }
    return {"ok": True, "events": [event], "sigma_pb": 1.5e3, "sigma_err_pb": 20.0, "n_accepted": 100, "seconds": 0.01}


def _parse():
    reply = synthetic_reply()
    raw = reply["events"][0]
    register_species(raw["species"])
    return event_from_reply(
        raw, seed=1, sqrt_s=91.2, beam_id="ee", process_id="ll_gmz_mu", title="e⁻ × e⁺ → γ*/Z → μ⁺μ⁻",
        sigma_pb=reply["sigma_pb"], sigma_err_pb=reply["sigma_err_pb"],
    )


def test_event_parses_with_beams_finals_and_decay_vertex():
    event = _parse()
    assert [p.pdg for p in event.beams()] == [11, -11]
    assert sorted(p.pdg for p in event.finals()) == [-211, -13, 13, 211]
    kaon = event.particles[6]
    assert event.decay_vertex_mm(kaon) == (10.0, 0.0, 0.0)
    # The primary angle is the first outgoing hard-process particle, the μ⁻.
    assert event.cos_theta == pytest.approx(10.0 / math.sqrt(20**2 + 30**2 + 10**2), rel=1e-3)


def test_conservation_uses_final_state_against_beams():
    report = conservation(_parse())
    assert report.ok
    assert report.delta_charge_thirds == 0
    assert abs(report.missing_px) < 1e-9 and abs(report.missing_py) < 1e-9


def test_species_registry_names_and_lifetimes():
    _parse()
    assert symbol(310) == "K_S0"
    assert english_name(310) == "neutral kaon (short-lived)"
    # Built-in PDG 2026 value wins for the charged pion; PYTHIA's table fills the kaon.
    assert proper_lifetime_s(211) == pytest.approx(7.8045 / 299_792_458.0)
    assert proper_lifetime_s(310) == pytest.approx(26.84e-3 / 299_792_458.0)


def test_traces_start_at_vertices_and_v0_is_dashed_to_its_decay():
    event = _parse()
    traces = build_traces(event, 3.8)
    kinds = sorted(trace.kind for trace in traces)
    assert kinds.count("decayed_neutral") == 1
    v0 = next(trace for trace in traces if trace.kind == "decayed_neutral")
    end = v0.points[-1]
    assert math.dist(end, (0.010, 0.0, 0.0)) < 1e-6
    pions = [trace for trace in traces if trace.kind == "track" and trace.label is None]
    assert len(pions) == 2
    assert all(math.dist(trace.points[0], (0.010, 0.0, 0.0)) < 1e-9 for trace in pions)
    muons = [trace for trace in traces if trace.label == "μ"]
    assert len(muons) == 2 and all(trace.deposit == "muon" for trace in muons)
    counts = count_summary(event)
    assert counts["charged"] == 4 and counts["leptons"] == 2


def test_worker_path_translates_to_wsl_and_follows_frozen_builds(monkeypatch):
    assert worker_path_in_wsl().startswith("/mnt/")
    assert worker_path_in_wsl().endswith("/hep/worker.py")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", r"C:\Temp\_MEI123", raising=False)
    assert worker_path_in_wsl() == "/mnt/c/Temp/_MEI123/hep/worker.py"


def test_pythia_panels_draw_a_full_event():
    from hadronica.app import LabApp

    app = LabApp(size=(1480, 900), headless=True, seed=7)
    try:
        app.engine_kind = "pythia"
        app.py_catalog = {"ll_gmz_mu": {"title": "γ*/Z → μ⁺μ⁻", "beams": "lepton"}}
        app.py_process = "ll_gmz_mu"
        config = app._py_config()
        app._py_accept(config, synthetic_reply(), "collide")
        assert app.report.ok
        for anim in (0.5, 3.0, 10.0):
            app.anim = anim
            for tab in ("summary", "products", "charts"):
                app.tab = tab
                app.draw()
        app.view_3d = True
        app.draw()
        app.show_report = True
        app.draw()
        app.show_report = False
        # Hover help for the PYTHIA-only controls.
        target = next(rect for rect, action, payload in app.hot if action == "pyprocess")
        app.pointer_pos = target.center
        app.view_3d = False
        app.draw()
        assert "leading-order" in app.guide_focus_body
        # With a Delphes reconstruction attached, the summary shows reconstructed objects.
        reply = synthetic_reply()
        reply["events"][0]["reco"] = {
            "detector": "ALEPH at LEP (Delphes)",
            "muons": [[44.0, 0.2, 2.1, -1.0], [43.5, -0.2, -1.0, 1.0]],
            "electrons": [], "photons": [], "jets": [], "towers": [[1.0, 0.1, 0.5, 1.0, 0.8, 0.2]],
            "met": [0.7, 1.0],
        }
        app._py_accept(config, reply, "collide")
        app.anim = 10.0
        app.tab = "summary"
        app.pointer_pos = (-1, -1)
        app.draw()
        assert app.event.reco["detector"].startswith("ALEPH")
    finally:
        pygame.quit()


def _wsl_engine_available() -> bool:
    if os.environ.get("HADRONICA_TEST_WSL", "1") == "0" or sys.platform != "win32":
        return False
    try:
        probe = subprocess.run(
            ["wsl.exe", "-d", engine.WSL_DISTRO, "--", "bash", "-lc", f"test -x {engine.ENV_PYTHON}"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not _wsl_engine_available(), reason="PYTHIA environment in WSL is not installed")
def test_real_pythia_engine_generates_conserving_events():
    import time

    bridge = PythiaEngine()
    bridge.start()
    try:
        deadline = time.time() + 300
        events = []
        bridge.request_events({"beams": "ee", "process": "ll_gmz_had", "sqrt_s": 91.1879, "seed": 5}, 40)
        bridge.request_events({"beams": "pp", "process": "pp_z_ll", "sqrt_s": 13600.0, "seed": 5, "detector": True,
                               "pileup": 30}, 5)
        while len(events) < 2 and time.time() < deadline:
            for kind, payload in bridge.poll():
                assert kind != "error", payload
                if kind == "events":
                    events.append(payload)
            time.sleep(0.05)
        assert len(events) == 2, bridge.message
        (config_z, reply_z), (config_pp, reply_pp) = events
        # Z → hadrons at the pole with QED ISR: σ⁰_had = 41.48 nb (PDG) less ~25 % from ISR, ≈ 30.5 nb;
        # the window allows the statistical error of 40 events.
        sigma_nb = reply_z["sigma_pb"] / 1.0e3
        assert 20.0 < sigma_nb < 40.0
        for raw in reply_z["events"] + reply_pp["events"]:
            register_species(raw["species"])
        parsed = [
            event_from_reply(raw, seed=5, sqrt_s=float(cfg["sqrt_s"]), beam_id=cfg["beams"], process_id=cfg["process"],
                             title="", sigma_pb=0.0, sigma_err_pb=0.0)
            for cfg, reply in ((config_z, reply_z), (config_pp, reply_pp))
            for raw in reply["events"]
        ]
        assert all(conservation(event).ok for event in parsed)
        pp_events = [event for event in parsed if event.beam_id == "pp"]
        assert all(event.reco and event.reco["detector"].startswith("CMS") for event in pp_events)
        # Pileup ⟨μ⟩ = 30: many reconstructed vertices, and pileup-flagged tracks.
        assert all(event.reco.get("n_vertices", 0) > 10 for event in pp_events)
        assert all(any(track[5] for track in event.reco["tracks_z"]) for event in pp_events)
    finally:
        bridge.stop()
