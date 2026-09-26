"""End-to-end self-test of the application: ``Hadronica.exe --selftest [report.json]``.

Runs headless inside the real application object, exactly as a user session
would: the built-in engine for every process at every preset energy, then the
PYTHIA engine started through the same bridge the GUI uses (from the files
bundled in a frozen build), in each of its main modes. Every event must
conserve four-momentum and charge, and every panel is drawn. The result goes
to a JSON report; the exit code is 0 only if every check passed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback

CHECK_TIMEOUT = 600.0  # seconds per PYTHIA request (first start compiles nothing, but loads PDFs)


class SelfTest:
    def __init__(self, app) -> None:
        self.app = app
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def draw_everything(self) -> None:
        app = self.app
        for tab in ("summary", "products", "charts"):
            app.tab = tab
            for view_3d in (False, True):
                app.view_3d = view_3d
                app.anim = 10.0
                app.draw()
        app.view_3d = False

    # -- built-in engine ---------------------------------------------------

    def builtin(self) -> None:
        from hadronica.app import PRESETS
        from hadronica.processes import BEAMS, all_processes

        app = self.app
        app._activate("engine", "builtin", (0, 0))
        failures, events = [], 0
        for beam_id in BEAMS:
            app._activate("beam", beam_id, (0, 0))
            for _label, energy, range_name in PRESETS:
                app.range_name = range_name
                app.set_energy(energy)
                for process in all_processes():
                    if not process.allowed(BEAMS[beam_id], app.sqrt_s):
                        continue
                    app._activate("process", process.id, (0, 0))
                    for _ in range(3):
                        app.spawn(animate=False)
                        events += 1
                        if not app.report.ok:
                            failures.append(f"{beam_id} {process.id} {energy:g} GeV")
                    self.draw_everything()
        self.check("built-in engine: every process at every preset conserves", not failures,
                   f"{events} events; failures: {failures[:5]}")

    # -- PYTHIA engine -----------------------------------------------------

    def wait_ready(self) -> bool:
        app = self.app
        deadline = time.time() + CHECK_TIMEOUT
        while time.time() < deadline:
            app._py_poll()
            if app.pythia is not None and app.pythia.state == "ready" and app.py_catalog:
                return True
            if app.pythia is not None and app.pythia.state == "error":
                return False
            time.sleep(0.05)
        return False

    def collide(self, name: str, beam: str, process: str, energy: float, **options) -> dict | None:
        """Generate one event through the GUI path; returns the config it used, or None on failure."""
        from hadronica.engine import PythiaEvent

        app = self.app
        app.py_beam = beam
        app.py_energy[beam] = energy
        app.py_process = process
        pileup = options.pop("pileup", 0)
        app.py_options.update({"detector": False, "nlo": True, "tune": True, **options})
        app.py_pileup = pileup
        config = app._py_config()
        app.event = None
        app.error = ""
        app._py_collide()
        deadline = time.time() + CHECK_TIMEOUT
        while time.time() < deadline and not isinstance(app.event, PythiaEvent) and not app.error:
            app._py_poll()
            time.sleep(0.05)
        event = app.event if isinstance(app.event, PythiaEvent) else None
        ok = event is not None and app.report is not None and app.report.ok
        detail = app.error or ("" if ok else "no event or conservation failed")
        if event is not None:
            self.draw_everything()
            detail = f"σ = {app.py_sigma.get(app._py_key(config), (0,))[0]:.4g} pb; {len(event.finals())} final particles"
            if config.get("detector"):
                ok = ok and bool(event.reco)
                detail += f"; detector {event.reco.get('detector', '?') if event.reco else 'missing'}"
        self.check(name, ok, detail)
        return config if ok else None

    def pythia(self) -> None:
        app = self.app
        app._activate("engine", "pythia", (0, 0))
        if not self.check("PYTHIA engine starts from the bundled worker", self.wait_ready(),
                          app.pythia.message if app.pythia else "no engine"):
            return
        self.check("PYTHIA catalog", len(app.py_catalog) >= 18, f"{len(app.py_catalog)} processes")
        self.collide("e⁺e⁻ → Z → hadrons at the Z pole", "ee", "ll_gmz_had", 91.1879)
        self.collide("e⁺e⁻ → ZH at 240 GeV with IDEA detector", "ee", "ll_zh", 240.0, detector=True)
        self.collide("μ⁺μ⁻ → t t̄ at 3 TeV with the muon-collider detector", "mumu", "ll_ttbar", 3000.0,
                     detector=True)
        self.collide("pp → H → 4ℓ at 13.6 TeV", "pp", "pp_h_4l", 13600.0)
        z_nlo = self.collide("pp → Z NLO (FxFx) with the Hadronica tune", "pp", "pp_z_ll", 13600.0)
        if app.py_nlo_samples.get("pp_z_ll@13600"):
            self.check("NLO mode used the MC@NLO sample and the tune", bool(z_nlo) and z_nlo.get("source") == "nlo"
                       and (z_nlo.get("tune") == "hadronica" or not app.py_tune), str(z_nlo))
        self.collide("pp → t t̄ NLO + MadSpin, CMS detector, pileup μ = 60", "pp", "pp_ttbar", 13600.0,
                     detector=True, pileup=60)

    def run(self) -> tuple[bool, float]:
        started = time.time()
        for part in (self.builtin, self.pythia):
            try:
                part()
            except Exception:  # noqa: BLE001 - every failure is a result, not a crash
                self.check(f"{part.__name__} raised", False, traceback.format_exc()[-1500:])
        self.app._shutdown_engine()
        passed = all(check["ok"] for check in self.checks)
        return passed, time.time() - started


def run_selftest(report_path: str) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from hadronica.app import LabApp

    app = LabApp(size=(1480, 900), headless=True, seed=11)
    test = SelfTest(app)
    passed, seconds = test.run()
    report = {
        "passed": passed,
        "seconds": round(seconds, 1),
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "platform": platform.platform(),
        "checks": test.checks,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    return 0 if passed else 1
