"""System tests: the whole application exercised the way a user drives it.

* The built-in engine over every beam, a dense set of energies (range edges and
  production thresholds included), ISR on and off, and every allowed process.
* Custom collisions of every species pair the picker offers, at random momenta
  and angles.
* Long randomized click sessions on the real interface, in both engines.
* The application self-test (``Hadronica.exe --selftest``) run in-process, and every
  PYTHIA process at every preset through the same bridge the GUI uses.

Every event shown must conserve four-momentum and charge, and nothing may raise.
"""

from __future__ import annotations

import math
import random
import time

import pygame
import pytest

from smlab.app import LabApp
from smlab.constants import M_H, M_T, M_Z
from smlab.engine import PythiaEngine, PythiaEvent, conservation, event_from_reply
from smlab.processes import BEAMS, all_processes
from smlab.report import register_species
from tests.test_engine import _wsl_engine_available

WSL = _wsl_engine_available()
live = pytest.mark.skipif(not WSL, reason="PYTHIA environment in WSL is not installed")


@pytest.fixture
def app():
    application = LabApp(size=(1480, 900), headless=True, seed=13)
    yield application
    application._shutdown_engine()
    pygame.quit()


def _draw_all(application) -> None:
    for tab in ("summary", "products", "charts"):
        application.tab = tab
        for view_3d in (False, True):
            application.view_3d = view_3d
            application.anim = 10.0
            application.draw()
    application.view_3d = False


# Range edges, the Z pole and its flanks, and just above each production threshold.
ENERGIES = sorted({0.4, 1.0, 3.0, 10.0, 30.0, 60.0, 88.0, M_Z, 94.0, 140.0, 150.0, 161.0,
                   M_Z + M_H + 0.5, 2 * M_T + 0.5, 250.0, 365.0, 500.0})


def test_builtin_engine_every_beam_energy_isr_and_process(app):
    failures, events = [], 0
    for beam_id, beam in BEAMS.items():
        app._activate("beam", beam_id, (0, 0))
        for isr in ((True, False) if beam_id in ("ee", "mumu") else (False,)):
            app.isr = isr
            for energy in ENERGIES:
                app.set_energy(energy)
                app.refresh_physics()
                for process, sigma in app.rows:
                    assert math.isfinite(sigma) and sigma >= 0.0, (beam_id, energy, process.id)
                    app._activate("process", process.id, (0, 0))
                    for _ in range(2):
                        app.spawn(animate=False)
                        events += 1
                        if not app.report.ok:
                            failures.append((beam_id, isr, energy, process.id))
                _draw_all(app)
    assert events > 1500
    assert not failures, failures[:10]


def _picker_species(application) -> list[int]:
    application._activate("beam", "custom", (0, 0))
    application.picker = "a"
    application.draw()
    species = sorted({int(payload[1]) for _rect, action, payload in application.hot if action == "species"})
    application.picker = None
    return species


def test_custom_collisions_of_every_offered_pair(app):
    """Supported exactly for the particle–antiparticle pairs of a modelled beam; √s is the invariant mass."""
    from smlab.particles import species as species_info

    rng = random.Random(4)
    offered = _picker_species(app)
    assert len(offered) >= 6
    beam_pairs = {frozenset((beam.pdg_plus, beam.pdg_minus)) for beam in BEAMS.values()}
    supported = 0
    for pdg_a in offered:
        for pdg_b in offered:
            app.pdg_a, app.pdg_b = pdg_a, pdg_b
            p_a, p_b = rng.uniform(1.0, 250.0), rng.uniform(1.0, 250.0)
            angle = rng.uniform(20.0, 180.0)
            app.momentum_a, app.momentum_b, app.collide_angle = p_a, p_b, angle
            app._apply_custom()
            # s = m_a² + m_b² + 2 (E_a E_b − p_a p_b cos θ), θ the angle between the two momenta.
            m_a, m_b = species_info(pdg_a).mass, species_info(pdg_b).mass
            e_a, e_b = math.hypot(p_a, m_a), math.hypot(p_b, m_b)
            s_inv = m_a**2 + m_b**2 + 2.0 * (e_a * e_b - p_a * p_b * math.cos(math.radians(angle)))
            assert app.sqrt_s == pytest.approx(math.sqrt(s_inv), rel=1e-9), (pdg_a, pdg_b)
            expected = frozenset((pdg_a, pdg_b)) in beam_pairs and app.sqrt_s >= 0.4
            assert app.custom_supported == expected, (pdg_a, pdg_b, app.sqrt_s)
            if not expected:
                assert app.event is None  # no fabricated event for an unmodelled pair
                continue
            supported += 1
            app.spawn(animate=False)
            assert app.report.ok, (pdg_a, pdg_b)
            app.draw()
    assert supported >= 2 * len(beam_pairs & {frozenset((a, b)) for a in offered for b in offered}) - 2


# Actions that would leave the app under test (start a validation run in WSL) or switch engine.
FUZZ_SKIP = {"val-run", "engine"}


def _fuzz(application, steps: int, rng: random.Random, pump=None) -> tuple[int, int]:
    clicks = shown = 0
    for _ in range(steps):
        application.anim += rng.choice((0.0, 0.3, 3.0, 10.0))
        application.draw()
        targets = [(rect, action) for rect, action, _payload in application.hot if action not in FUZZ_SKIP]
        roll = rng.random()
        if roll < 0.85 and targets:
            rect, _action = rng.choice(targets)
            point = (rng.randint(rect.left, max(rect.left, rect.right - 1)),
                     rng.randint(rect.top, max(rect.top, rect.bottom - 1)))
            application._handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point))
            application._handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=point))
            clicks += 1
        elif roll < 0.93:
            application._handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=rng.choice((-3, -1, 1, 3))))
        else:
            application._handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                                   pos=(rng.randrange(1480), rng.randrange(900))))
            application._handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1,
                                                   pos=(rng.randrange(1480), rng.randrange(900))))
        application._advance_batch()
        if pump is not None:
            pump()
        if application.event is not None and application.report is not None:
            shown += 1
            assert application.report.ok, "a displayed event does not conserve"
    return clicks, shown


def test_random_session_on_the_builtin_engine(app):
    clicks, shown = _fuzz(app, 2500, random.Random(20260924))
    assert clicks > 1500 and shown > 500


@live
def test_application_selftest_passes_in_process(app):
    from smlab.selftest import SelfTest

    test = SelfTest(app)
    passed, _seconds = test.run()
    assert passed, [check for check in test.checks if not check["ok"]]


LEPTON_ENERGIES = (M_Z, 161.0, 240.0, 365.0, 1000.0, 3000.0)
PP_ENERGIES = (900.0, 7000.0, 13000.0, 13600.0, 14000.0)


@live
def test_every_pythia_process_at_every_preset():
    from smlab.app_pythia import PY_THRESHOLDS

    bridge = PythiaEngine()
    bridge.start()
    try:
        deadline = time.time() + 300
        catalog = {}
        while not catalog and time.time() < deadline:
            for kind, payload in bridge.poll():
                assert kind != "error", payload
                if kind == "ready":
                    catalog = payload["processes"]
            time.sleep(0.05)
        assert len(catalog) >= 18
        configs = []
        for key, spec in catalog.items():
            if spec["beams"] == "pp":
                configs += [{"beams": "pp", "process": key, "sqrt_s": e} for e in PP_ENERGIES]
            else:
                for beams in ("ee", "mumu"):
                    configs += [{"beams": beams, "process": key, "sqrt_s": e}
                                for e in LEPTON_ENERGIES if e >= PY_THRESHOLDS.get(key, 0.0)]
        for index, config in enumerate(configs):
            bridge.request_events({**config, "seed": 100 + index}, 2)
        replies, deadline = [], time.time() + 1800
        while len(replies) < len(configs) and time.time() < deadline:
            for kind, payload in bridge.poll():
                assert kind != "error", payload
                if kind == "events":
                    replies.append(payload)
            time.sleep(0.02)
        assert len(replies) == len(configs)
        bad = []
        for config, reply in replies:
            assert reply["sigma_pb"] > 0.0 and math.isfinite(reply["sigma_pb"]), config
            for raw in reply["events"]:
                register_species(raw["species"])
                event = event_from_reply(raw, seed=1, sqrt_s=float(config["sqrt_s"]), beam_id=config["beams"],
                                         process_id=config["process"], title="", sigma_pb=0.0, sigma_err_pb=0.0)
                if not conservation(event).ok:
                    bad.append(config)
        assert not bad, bad[:5]
    finally:
        bridge.stop()


@live
def test_random_session_on_the_pythia_engine(app):
    app._activate("engine", "pythia", (0, 0))
    deadline = time.time() + 300
    while not (app.pythia and app.pythia.state == "ready" and app.py_catalog) and time.time() < deadline:
        app._py_poll()
        time.sleep(0.05)
    assert app.py_catalog, app.pythia.message if app.pythia else "no engine"

    def pump():
        app._py_poll()

    clicks, _shown = _fuzz(app, 400, random.Random(7), pump=pump)
    # Let queued requests finish, then everything shown must still be consistent.
    deadline = time.time() + 600
    while app.py_expect and time.time() < deadline:
        app._py_poll()
        time.sleep(0.05)
    assert not app.py_expect, "requests never answered"
    assert clicks > 250
    assert app.pythia.state in ("ready", "busy")
    if isinstance(app.event, PythiaEvent):
        assert conservation(app.event).ok
