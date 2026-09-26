"""PYTHIA 8.3 mode for the Hadronica window.

Mixed into :class:`hadronica.app.LabApp`. Everything here drives the WSL worker
through :mod:`hadronica.engine` and presents full events: showers, hadrons,
jets, displaced vertices, and PYTHIA's own cross sections.
"""

from __future__ import annotations

import math

import pygame

from hadronica.constants import M_H, M_T, M_Z, format_cross_section, format_energy
from hadronica.engine import PythiaEngine, PythiaEvent, conservation, event_from_reply
from hadronica.fullscene import MIN_PT_CHARGED, build_traces, count_summary
from hadronica.guide import GuideCard
from hadronica.histogram import Histogram
from hadronica.lorentz import FourVector
from hadronica.report import brief_result, english_name, register_species, rest_mass, symbol
from hadronica.theme import ACCENT, ACCENT_SOFT, BG, BORDER, SURFACE_2, SURFACE_3, TEXT, TEXT_2, TEXT_3, WARN, particle_color

PY_BEAMS = (("ee", "e⁻ × e⁺"), ("mumu", "μ⁻ × μ⁺"), ("pp", "p × p"))
PY_RANGES = {
    "Z": (60.0, 140.0),
    "high": (140.0, 500.0),
    "TeV": (500.0, 3000.0),
    "pp": (900.0, 14000.0),
}
PY_PRESETS = {
    "lepton": (
        ("Z pole", M_Z, "Z"),
        ("WW 161", 161.0, "high"),
        ("ZH 240", 240.0, "high"),
        ("t t̄ 365", 365.0, "high"),
        ("1 TeV", 1000.0, "TeV"),
        ("3 TeV", 3000.0, "TeV"),
    ),
    "pp": (
        ("900 GeV", 900.0, "pp"),
        ("7 TeV", 7000.0, "pp"),
        ("13 TeV", 13000.0, "pp"),
        ("13.6 TeV", 13600.0, "pp"),
        ("14 TeV", 14000.0, "pp"),
    ),
}
# Lowest √s at which each lepton-collider process is offered.
PY_THRESHOLDS = {
    "ll_ww": 150.0,
    "ll_zz": 2.0 * M_Z - 10.0,
    "ll_zh": M_Z + M_H + 1.0,
    "ll_vbf_h": M_H + 20.0,
    "ll_ttbar": 2.0 * M_T + 1.0,
}
BATCH_CHUNK = 25

PY_HELP = {
    "engine": (
        "Physics engine",
        "Built-in LO is Hadronica's own leading-order generator: instant, exact formulas, partons drawn as lines. "
        "PYTHIA 8.3 is the research-grade generator used at the LHC: QED and QCD showers, multiparton interactions, "
        "Lund string hadronization, and hadron decays with real decay vertices, with the Monash 2013 tune. "
        "It runs in WSL, so the first event after a change takes a moment to initialize.",
    ),
    "pybeam": (
        "Colliding particles",
        "Electron–positron and muon–antimuon beams radiate photons through PYTHIA's QED structure functions. "
        "Proton–proton collisions use the NNPDF2.3 QCD+QED parton distributions of the Monash tune, "
        "plus multiparton interactions and beam remnants: this is how LHC events are simulated.",
    ),
    "pyprocess": (
        "Hard process",
        "The partonic process PYTHIA generates; everything else in the event (radiation, the underlying event, "
        "hadrons, decays) is added by PYTHIA around it. The rate is PYTHIA's leading-order cross section, "
        "estimated from the events generated so far, with its statistical uncertainty. Measured LHC rates are "
        "often higher, because higher-order corrections are not included (for t t̄ at 13.6 TeV, about 1.4–1.5×).",
    ),
    "pytoggle:isr": (
        "Initial-state radiation",
        "PYTHIA gives each lepton a QED parton distribution and showers photons off the beams. Off removes it, "
        "so the collision uses the full beam energy. It changes the rate on and near the Z peak.",
    ),
    "pytoggle:mpi": (
        "Multiparton interactions",
        "In a proton–proton collision several parton pairs scatter at once. They produce the underlying event: "
        "extra soft tracks throughout the detector. Turn it off to see only the hard scattering and its radiation.",
    ),
    "pytoggle:hadronize": (
        "Hadronization",
        "Quarks and gluons turn into hadrons through the Lund string model. Off leaves the partons after the shower; "
        "they are then not drawn, because free quarks never reach a detector.",
    ),
    "pytoggle:detector": (
        "Detector simulation",
        "Runs Delphes 3.5, the standard fast simulation of a collider detector, on each event, with the card for "
        "this collider: CMS for proton beams, ALEPH at LEP energies, IDEA at FCC-ee, CLICdet above 400 GeV, and the "
        "muon-collider detector. It applies measured resolutions and efficiencies and reconstructs electrons, muons, "
        "photons, jets with b-tagging, and missing momentum, which then replace the generator-level objects on "
        "screen. Each batch of events takes a few extra seconds.",
    ),
    "pytoggle:nlo": (
        "NLO hard process",
        "Uses events from MadGraph5_aMC@NLO 3.5 at next-to-leading order in QCD, matched to the PYTHIA shower "
        "with the MC@NLO method and MadGraph's own PYTHIA 8 matching settings, with NNPDF3.1 NLO parton "
        "distributions. The cross section is the NLO value with its scale uncertainty. About a fifth of MC@NLO "
        "events carry a negative weight; distributions use the signed weights. Validated against LHC data: NLO "
        "improves the CMS t t̄ distributions (χ²/ndf 7.4 → 2.2), but the ATLAS Z transverse-momentum shape is described "
        "better by PYTHIA's tuned leading order (4.2 against 25.8): in an inclusive NLO Z sample the high-pT tail is "
        "only leading-order Z + 1 jet, and the low-pT shape depends on the shower settings. Available where a "
        "sample has been generated (hep/mg5/generate_nlo.sh).",
    ),
    "pyenergy": (
        "Collision energy",
        "The center-of-mass energy of the two beams. For protons it is shared among the partons, so each hard "
        "collision has its own lower energy √ŝ, shown in the summary.",
    ),
}


class PythiaMode:
    """State and panels for the PYTHIA engine. Expects the attributes and widgets of LabApp."""

    def _pythia_init_state(self) -> None:
        self.engine_kind = "builtin"
        self.pythia: PythiaEngine | None = None
        self.py_catalog: dict[str, dict] = {}
        self.py_beam = "ee"
        self.py_process = "ll_gmz_had"
        self.py_energy = {"ee": M_Z, "mumu": M_Z, "pp": 13600.0}
        self.py_range = {"ee": "Z", "mumu": "Z", "pp": "pp"}
        self.py_options = {"isr": True, "mpi": True, "hadronize": True, "detector": False, "nlo": True,
                           "tune": True}
        self.py_pileup = 0
        self.py_nlo_samples: dict[str, dict] = {}
        self.py_last_nlo: dict | None = None
        self.py_tune: dict | None = None
        self.py_sigma: dict[tuple, tuple[float, float, int]] = {}
        self.py_expect: list[str] = []
        self.py_batch_left = 0
        self.py_event_count = 0
        self.py_shat: list[float] = []
        self.py_mult: list[int] = []
        self._py_traces_key = None
        self._py_traces = []

    # -- engine --------------------------------------------------------------

    @property
    def pythia_mode(self) -> bool:
        return self.engine_kind == "pythia"

    def _py_ensure_engine(self) -> None:
        if self.pythia is None:
            self.pythia = PythiaEngine()
        if self.pythia.state in ("offline", "error"):
            self.pythia.start()

    def _py_config(self) -> dict:
        beams = self.py_beam
        config = {
            "beams": beams,
            "process": self.py_process,
            "sqrt_s": round(self.py_energy[beams], 6),
            "seed": int(self.base_seed) % 900000000,
            "hadronize": self.py_options["hadronize"],
            "detector": self.py_options["detector"],
        }
        if beams == "pp":
            config["mpi"] = self.py_options["mpi"]
            if self.py_options["detector"] and self.py_pileup:
                config["pileup"] = self.py_pileup
            if self.py_options["nlo"] and self._py_nlo_sample() is not None:
                config["source"] = "nlo"
                if self.py_options["tune"] and self._py_tune_applies():
                    config["tune"] = "hadronica"
        else:
            config["isr"] = self.py_options["isr"]
        return config

    def _py_nlo_sample(self) -> dict | None:
        """The MC@NLO sample for the selected process and energy, if one was generated."""
        if self.py_beam != "pp":
            return None
        return self.py_nlo_samples.get(f"{self.py_process}@{int(round(self.py_energy['pp']))}")

    def _py_tune_applies(self) -> bool:
        """Whether the Hadronica shower tune (hep/tune.py) exists for the selected MC@NLO sample."""
        sample = self._py_nlo_sample()
        if not self.py_tune or not self.py_tune.get("settings") or sample is None:
            return False
        applies = self.py_tune.get("applies_to")
        return not applies or sample.get("sample") in applies

    @staticmethod
    def _py_key(config: dict) -> tuple:
        return tuple(sorted((k, v) for k, v in config.items() if k != "seed"))

    def _py_kind(self) -> str:
        return "pp" if self.py_beam == "pp" else "lepton"

    def _py_processes(self) -> list[tuple[str, str]]:
        kind = self._py_kind()
        energy = self.py_energy[self.py_beam]
        out = []
        for key, spec in self.py_catalog.items():
            if spec["beams"] != kind:
                continue
            if energy < PY_THRESHOLDS.get(key, 0.0):
                continue
            out.append((key, spec["title"]))
        return out

    def _py_ensure_process(self) -> None:
        options = [key for key, _title in self._py_processes()]
        if options and self.py_process not in options:
            self.py_process = options[0]

    def _py_request(self, n: int, tag: str) -> None:
        self._py_ensure_engine()
        self._py_ensure_process()
        self.pythia.request_events(self._py_config(), n)
        self.py_expect.append(tag)

    def _py_collide(self) -> None:
        if self.pythia is not None and self.pythia.state == "error":
            self.pythia = None
        self._py_request(1, "collide")

    def _py_batch(self) -> None:
        self.py_batch_left = 200
        self.batch_total = 200
        self.batch_left = 200
        for _ in range(200 // BATCH_CHUNK):
            self._py_request(BATCH_CHUNK, "batch")

    def _py_poll(self) -> None:
        if self.pythia is None:
            return
        for kind, payload in self.pythia.poll():
            if kind == "ready":
                self.py_catalog = dict(payload.get("processes", {}))
                self.py_nlo_samples = dict(payload.get("nlo_samples", {}))
                self.py_tune = payload.get("tune")
                self._py_ensure_process()
                self.error = ""
            elif kind == "error":
                if self.py_expect:
                    self.py_expect.pop(0)
                self.error = str(payload)
                self.batch_left = 0
            elif kind == "events":
                config, reply = payload
                tag = self.py_expect.pop(0) if self.py_expect else "collide"
                self._py_accept(config, reply, tag)

    def _py_accept(self, config: dict, reply: dict, tag: str) -> None:
        sigma = float(reply.get("sigma_pb", 0.0))
        err = float(reply.get("sigma_err_pb", 0.0))
        self.py_sigma[self._py_key(config)] = (sigma, err, int(reply.get("n_accepted", 0)))
        self.py_last_nlo = reply.get("nlo")
        title_beam = dict(PY_BEAMS)[config["beams"]]
        title = f"{title_beam} → {self.py_catalog.get(config['process'], {}).get('title', config['process'])}"
        last = None
        for raw in reply["events"]:
            register_species(raw.get("species", {}))
            self.py_event_count += 1
            event = event_from_reply(
                raw,
                seed=int(config["seed"]),
                sqrt_s=float(config["sqrt_s"]),
                beam_id=config["beams"],
                process_id=config["process"],
                title=title,
                sigma_pb=sigma,
                sigma_err_pb=err,
            )
            # MC@NLO events carry signed weights; histograms must use them.
            weight = float(raw.get("weight", 1.0)) if config.get("source") == "nlo" else 1.0
            event.weight = weight
            self.py_shat.append((event.sqrt_s_hat, weight))
            self.py_mult.append((sum(1 for p in event.finals() if abs(p.charge) > 1.0e-6), weight))
            last = event
        if last is None:
            return
        self.event = last
        self.report = conservation(last)
        self.error = ""
        self.scroll_tree = 0
        if tag == "batch":
            self.batch_left = max(0, self.batch_left - len(reply["events"]))
            self.anim = 10.0
        else:
            self.anim = 0.0

    # -- left panel ----------------------------------------------------------

    def _engine_switch(self, x: int, y: int, w: int) -> int:
        """Segmented engine switch; the engine state is written just to its right."""
        rect = pygame.Rect(x, y, w, 32)
        self._segmented(
            rect,
            [("Built-in LO", "builtin", not self.pythia_mode), ("PYTHIA 8.3", "pythia", self.pythia_mode)],
            "engine",
        )
        if self.pythia_mode and self.pythia is not None:
            state = {"starting": "starting…", "busy": "working…", "ready": "ready", "error": "unavailable"}.get(
                self.pythia.state, "")
            color = WARN if self.pythia.state == "error" else TEXT_3
            self._blit("caption", state, color, (rect.right + 10, rect.centery), "midleft")
        return rect.bottom

    def _py_setup_particles(self, x: int, y: int, w: int) -> int:
        detail = "hadron collider" if self.py_beam == "pp" else "lepton collider"
        y = self._step_header(x, y, w, "1", "Colliding particles", detail)
        return self._chips(x, y, w, [(label, key, key == self.py_beam) for key, label in PY_BEAMS], "pybeam")

    def _py_setup_energy(self, x: int, y: int, w: int) -> int:
        y = self._step_header(x, y, w, "2", "Collision energy", "√s, center of mass")
        energy = self.py_energy[self.py_beam]
        box = pygame.Rect(x, y, w, 48)
        hover = self._hovered(box)
        pygame.draw.rect(self.screen, SURFACE_3 if hover else SURFACE_2, box, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT if self.energy_focus else BORDER, box, 1, border_radius=10)
        shown = self.energy_buffer if self.energy_focus else (f"{energy:.3f}" if energy < 1000 else f"{energy:.1f}")
        number = self._blit("big", shown, TEXT, (box.x + 14, box.centery), "midleft")
        self._blit("body", "GeV", TEXT_2, (number.right + 10, box.centery + 3), "midleft")
        self._blit("caption", "type, then Enter" if self.energy_focus else "click to type", TEXT_3, (box.right - 12, box.centery), "midright")
        self._hit(box, "energy")
        y = box.bottom + 16
        lo, hi = PY_RANGES[self.py_range[self.py_beam]]
        self._energy_rect = pygame.Rect(x + 4, y, w - 8, 18)
        self._slider(self._energy_rect, (energy - lo) / (hi - lo), "pyslider")
        y += 22
        self._blit("mono_small", f"{lo:g}", TEXT_3, (x, y))
        self._blit("mono_small", f"{hi:g} GeV", TEXT_3, (x + w, y), "topright")
        y += 20
        presets = PY_PRESETS[self._py_kind()]
        return self._chips(
            x, y, w,
            [(label, (label, value, window), abs(energy - value) < 0.02) for label, value, window in presets],
            "pypreset",
            height=26,
        )

    def _py_setup_final_state(self, x: int, y: int, w: int, soft_bottom: int) -> int:
        nlo_on = self.py_beam == "pp" and self.py_options["nlo"] and any(
            k.endswith(f"@{int(round(self.py_energy['pp']))}") for k in self.py_nlo_samples)
        y = self._step_header(x, y, w, "3", "Hard process", "σ: NLO where available" if nlo_on else "σ: PYTHIA LO")
        processes = self._py_processes()
        if not processes:
            message = self.pythia.message if self.pythia is not None else "Starting PYTHIA…"
            return self._wrap_block(x, y, w, message, "small", TEXT_3)
        row_h = 34
        full = len(processes) * row_h + 8
        list_h = int(max(136, min(full, soft_bottom - y)))
        rect = pygame.Rect(x, y, w, list_h)
        self.zone_process = rect.clip(self._clip) if self._clip is not None else rect
        pygame.draw.rect(self.screen, BG, rect, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=10)
        saved_clip = self._clip
        inner = rect.inflate(-4, -4)
        view = inner.clip(saved_clip) if saved_clip is not None else inner
        previous = self.screen.get_clip()
        self.screen.set_clip(view)
        self._clip = view
        self.scroll_proc = min(self.scroll_proc, max(0, full - list_h))
        row_y = rect.y + 4 - self.scroll_proc
        config = self._py_config()
        for key, title in processes:
            row = pygame.Rect(rect.x + 4, row_y, rect.w - 8, row_h - 2)
            selected = key == self.py_process
            hover = self._hovered(row)
            if selected:
                pygame.draw.rect(self.screen, ACCENT_SOFT, row, border_radius=7)
                pygame.draw.rect(self.screen, ACCENT, pygame.Rect(row.x, row.y + 6, 3, row.h - 12), border_radius=2)
            elif hover:
                pygame.draw.rect(self.screen, SURFACE_2, row, border_radius=7)
            row_config = {**config, "process": key}
            nlo_sample = self.py_nlo_samples.get(f"{key}@{int(round(self.py_energy['pp']))}") if self.py_beam == "pp" else None
            if nlo_sample and self.py_options["nlo"] and nlo_sample.get("sigma_pb") and not nlo_sample.get("fxfx"):
                # MadGraph's NLO cross section for the whole sample, not PYTHIA's running estimate.
                value_text = format_cross_section(nlo_sample["sigma_pb"]) + " NLO"
            elif nlo_sample and self.py_options["nlo"]:
                # FxFx: MadGraph's σ is before merging; PYTHIA's estimate after the veto is the physical one.
                known = self.py_sigma.get(self._py_key(row_config))
                value_text = (format_cross_section(known[0]) + " NLO") if known else "NLO"
            else:
                row_config.pop("source", None)
                known = self.py_sigma.get(self._py_key(row_config))
                value_text = format_cross_section(known[0]) if known else "—"
            value = self.text.render("mono_small", value_text, TEXT if selected else TEXT_3)
            label = self.text.fit("body", title, row.w - value.get_width() - 28)
            self._blit("body", label, TEXT if (selected or hover) else TEXT_2, (row.x + 12, row.centery), "midleft")
            self.screen.blit(value, value.get_rect(midright=(row.right - 10, row.centery)))
            self._hit(row, "pyprocess", key)
            row_y += row_h
        self._clip = saved_clip
        self.screen.set_clip(previous)
        return rect.bottom

    def _py_setup_options(self, x: int, y: int, w: int) -> int:
        y = self._step_header(x, y, w, "4", "Physics & detector")
        if self.py_beam == "pp":
            options = (("Multiparton interactions", "mpi"), ("Hadronization", "hadronize"))
        else:
            options = (("Initial-state radiation", "isr"), ("Hadronization", "hadronize"))
        if self._py_nlo_sample() is not None:
            options += (("NLO hard process (MadGraph5_aMC@NLO)", "nlo"),)
            if self.py_options["nlo"] and self._py_tune_applies():
                options += ((f"{self.py_tune.get('name') or 'Hadronica'} shower tune", "tune"),)
        options += (("Detector simulation (Delphes)", "detector"),)
        for label, key in options:
            self._switch_py(pygame.Rect(x - 8, y, w + 16, 32), label, self.py_options[key], key)
            y += 33
        if self.py_beam == "pp" and self.py_options["detector"]:
            self._blit("small", "Mean pileup μ", TEXT_2, (x, y + 4))
            self._blit("caption", "Run 3 ≈ 60 · HL-LHC 140", TEXT_3, (x + w, y + 6), "topright")
            y += 24
            self._segmented(
                pygame.Rect(x, y, w, 30),
                [(label, value, self.py_pileup == value) for label, value in
                 (("off", 0), ("30", 30), ("60", 60), ("140", 140))],
                "pypileup",
            )
            y += 38
        y += 6
        self._blit("small", "Solenoid field", TEXT_2, (x, y))
        self._blit("mono_small", f"{self.b_field:.2f} T", TEXT, (x + w, y + 2), "topright")
        self._field_rect = pygame.Rect(x + 4, y + 22, w - 8, 18)
        self._slider(self._field_rect, self.b_field / 4.0, "bslider")
        return y + 44

    def _switch_py(self, rect: pygame.Rect, label: str, value: bool, key: str) -> None:
        hover = self._hovered(rect)
        if hover:
            pygame.draw.rect(self.screen, SURFACE_2, rect, border_radius=8)
        self._blit("body", label, TEXT if hover else TEXT_2, (rect.x + 8, rect.centery), "midleft")
        track = pygame.Rect(rect.right - 46, rect.centery - 11, 38, 22)
        pygame.draw.rect(self.screen, ACCENT if value else SURFACE_3, track, border_radius=11)
        knob_x = track.right - 11 if value else track.x + 11
        pygame.draw.circle(self.screen, (8, 14, 26) if value else TEXT_2, (knob_x, track.centery), 8)
        self._hit(rect, "pytoggle", key)

    # -- actions ---------------------------------------------------------------

    def _py_activate(self, action: str, payload, pos) -> bool:
        if action == "engine":
            self.engine_kind = str(payload)
            self.event = None
            self.report = None
            self.error = ""
            self.scroll_proc = 0
            if self.pythia_mode:
                self._py_ensure_engine()
            else:
                self.refresh_physics()
                self.spawn()
            return True
        if action == "pybeam":
            self.py_beam = str(payload)
            self.scroll_proc = 0
            self.py_shat.clear()
            self.py_mult.clear()
            self._py_ensure_process()
            return True
        if action == "pyprocess":
            self.py_process = str(payload)
            self.py_shat.clear()
            self.py_mult.clear()
            return True
        if action == "pytoggle":
            self.py_options[str(payload)] = not self.py_options[str(payload)]
            return True
        if action == "pypreset":
            _label, value, window = payload
            self.py_range[self.py_beam] = window
            self._py_set_energy(value)
            return True
        if action == "pypileup":
            self.py_pileup = int(payload)
            return True
        if action == "pyslider":
            self.drag = "pyenergy"
            self._drag(pos)
            return True
        return False

    def _py_set_energy(self, value: float) -> None:
        if self.py_beam == "pp":
            lo, hi = PY_RANGES["pp"]
            value = min(hi, max(lo, value))
        else:
            value = min(3000.0, max(20.0, value))
            window = "Z" if value < 140.0 else ("high" if value <= 500.0 else "TeV")
            if not (PY_RANGES[self.py_range[self.py_beam]][0] <= value <= PY_RANGES[self.py_range[self.py_beam]][1]):
                self.py_range[self.py_beam] = window
        if abs(value - self.py_energy[self.py_beam]) > 1.0e-6:
            self.py_shat.clear()
            self.py_mult.clear()
        self.py_energy[self.py_beam] = value
        self._py_ensure_process()

    def _py_drag_energy(self, pos) -> None:
        lo, hi = PY_RANGES[self.py_range[self.py_beam]]
        t = (pos[0] - self._energy_rect.x) / max(1, self._energy_rect.w)
        value = lo + min(1.0, max(0.0, t)) * (hi - lo)
        self.py_energy[self.py_beam] = value
        self._py_ensure_process()

    # -- stage -----------------------------------------------------------------

    def _py_traces_for(self, event):
        key = (id(event), round(self.b_field, 4))
        if key != self._py_traces_key:
            self._py_traces = build_traces(event, self.b_field)
            self._py_traces_key = key
        return self._py_traces

    def _py_approach_pair(self):
        beams = {"ee": (11, -11), "mumu": (13, -13), "pp": (2212, 2212)}[self.py_beam]
        energy = self.py_energy[self.py_beam] / 2.0
        mass = rest_mass(beams[0]) if beams[0] != 2212 else 0.93827
        momentum = math.sqrt(max(energy * energy - mass * mass, 0.0))
        a = FourVector(energy, momentum, 0.0, 0.0)
        b = FourVector(energy, -momentum, 0.0, 0.0)
        return a, b, beams[0], beams[1], 180.0

    def _py_stage_title(self) -> tuple[str, str]:
        title = self.py_catalog.get(self.py_process, {}).get("title", "PYTHIA 8.3")
        beam = dict(PY_BEAMS)[self.py_beam]
        state = ""
        if self.pythia is not None and self.pythia.state in ("starting", "busy"):
            state = f"   ·   {self.pythia.message}"
        return f"{beam} → {title}", f"√s = {format_energy(self.py_energy[self.py_beam])}{state}"

    # -- results ----------------------------------------------------------------

    def _py_tab_summary(self, rect: pygame.Rect) -> None:
        event = self.event if isinstance(self.event, PythiaEvent) else None
        _approach, product = self._anim_phase()
        revealed = event is not None and product > 0.0
        config = self._py_config()
        config_sigma = self.py_sigma.get(self._py_key(config))
        sample = self._py_nlo_sample() if config.get("source") == "nlo" else None
        if sample is not None and sample.get("fxfx") and config_sigma:
            sigma_text = format_cross_section(config_sigma[0])
            sigma_sub = f"NLO FxFx-merged · ± {format_cross_section(config_sigma[1])}"
        elif sample is not None and sample.get("sigma_pb") and not sample.get("fxfx"):
            sigma_text = format_cross_section(sample["sigma_pb"])
            up, down = sample.get("scale_up_pct"), sample.get("scale_down_pct")
            sigma_sub = f"NLO · scale +{up:.1f}/−{down:.1f} %" if up is not None and down is not None else "NLO (MC@NLO)"
        elif config_sigma:
            sigma_text = format_cross_section(config_sigma[0])
            sigma_sub = f"LO · ± {format_cross_section(config_sigma[1])} · {config_sigma[2]} ev"
        else:
            sigma_text, sigma_sub = "—", "after the first event"
        if revealed:
            counts = count_summary(event)
            met = math.hypot(self.report.missing_px, self.report.missing_py)
            displaced = _displaced_vertices(event)
            lead = f"leading {event.jets[0].pt:.0f} GeV" if event.jets and event.beam_id == "pp" else (
                "Durham, y_cut 0.005" if event.beam_id != "pp" else "anti-kT R 0.4, pT > 20")
            jets_value, jets_sub = str(len(event.jets)), lead
            met_value, met_sub = f"{met:.1f} GeV", "neutrinos and losses"
            if event.reco:
                reco_jets = event.reco.get("jets", [])
                tagged = sum(1 for j in reco_jets if int(j[4]) & 1)
                jets_value, jets_sub = str(len(reco_jets)), f"reconstructed · {tagged} b-tagged"
                if "met" in event.reco:
                    met_value = f"{event.reco['met'][0]:.1f} GeV"
                    met_sub = f"reconstructed · truth {met:.1f}"
            tiles = (
                ("Hard scale √ŝ", format_energy(event.sqrt_s_hat), self.text.fit("caption", event.name, 140)),
                ("Cross section", sigma_text, sigma_sub),
                ("Charged particles", str(counts["charged"]), f"{len(event.finals())} final particles"),
                ("Jets", jets_value, jets_sub),
                ("Missing pT", met_value, met_sub),
                ("Displaced vertices", str(displaced), "decays > 0.5 mm from birth"),
            )
            if event.reco and event.reco.get("n_vertices"):
                pileup_tracks = sum(1 for t in event.reco.get("tracks_z", []) if t[5])
                tiles = tiles[:5] + (
                    ("Collision vertices", str(event.reco["n_vertices"]), f"{pileup_tracks} pileup tracks"),
                )
        else:
            waiting = "colliding" if event is not None else ("waiting" if self.py_expect else "no event yet")
            tiles = (
                ("Hard scale √ŝ", "…" if event else "—", waiting),
                ("Cross section", sigma_text, sigma_sub),
                ("Charged particles", "—", ""),
                ("Jets", "—", ""),
                ("Missing pT", "—", ""),
                ("Displaced vertices", "—", ""),
            )
        tile_w = (rect.w - 8) // 2
        tile_h = 58
        for index, (label, value, sub) in enumerate(tiles):
            col, row = index % 2, index // 2
            tile = pygame.Rect(rect.x + col * (tile_w + 8), rect.y + row * (tile_h + 8), tile_w, tile_h)
            pygame.draw.rect(self.screen, SURFACE_2, tile, border_radius=10)
            self._blit("caption", label.upper(), TEXT_3, (tile.x + 12, tile.y + 8))
            self._blit("h2", self.text.fit("h2", value, tile.w - 24), TEXT, (tile.x + 12, tile.y + 22))
            self._blit("caption", self.text.fit("caption", sub, tile.w - 24), TEXT_3, (tile.x + 12, tile.y + 42))
        y = rect.y + 3 * (tile_h + 8) + 8
        if not revealed:
            text = (
                "PYTHIA generates the full event: the hard process, radiation, the underlying event, hadrons, "
                "and their decays. Press Collide."
                if event is None
                else "The beams are approaching. The full event appears when they meet."
            )
            if self.error:
                text = self.error
            self._wrap_block(rect.x, y, rect.w, text, "small", WARN if self.error else TEXT_2)
            return
        self._blit("caption", "HARD PROCESS", TEXT_3, (rect.x, y))
        link = pygame.Rect(0, y - 5, 118, 24)
        link.right = rect.right
        self._button(link, "Full report", "report", kind="ghost")
        y += 24
        incoming, middle, outgoing = _hard_summary(event)
        lines = []
        if incoming:
            lines.append(("in", " ".join(incoming)))
        if middle:
            lines.append(("via", " ".join(middle)))
        if outgoing:
            lines.append(("out", " ".join(outgoing)))
        counts = count_summary(event)
        lines.append(("seen", f"{counts['charged']} charged, {counts['photons']} photons, {counts['neutral_hadrons']} neutral hadrons"))
        if counts["neutrinos"]:
            lines.append(("unseen", f"{counts['neutrinos']} neutrinos"))
        if event.reco:
            reco = event.reco
            found = []
            for key, name in (("electrons", "e"), ("muons", "μ"), ("photons", "γ")):
                items = reco.get(key, [])
                if items:
                    found.append(f"{len(items)} {name} ({', '.join(f'{item[0]:.0f}' for item in items[:3])} GeV)")
            jets = reco.get("jets", [])
            found.append(f"{len(jets)} jets")
            lines.append(("det", reco.get("detector", "Delphes")))
            lines.append(("reco", "; ".join(found)))
        box = pygame.Rect(rect.x, y, rect.w, 14 + 22 * len(lines))
        pygame.draw.rect(self.screen, SURFACE_2, box, border_radius=10)
        ly = box.y + 8
        for key, text in lines:
            self._blit("caption", key.upper(), TEXT_3, (box.x + 12, ly + 3))
            self._blit("body", self.text.fit("body", text, box.w - 70), TEXT, (box.x + 58, ly))
            ly += 22

    def _py_tab_products(self, rect: pygame.Rect) -> None:
        event = self.event if isinstance(self.event, PythiaEvent) else None
        if event is None:
            self._wrap_block(rect.x, rect.y, rect.w, "The event record appears here after a collision.", "small", TEXT_2)
            return
        self.zone_tree = rect
        pygame.draw.rect(self.screen, BG, rect, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=10)
        self._hit(rect, "help-tree")
        rows: list[tuple[str, object]] = [("head", "HARD PROCESS")]
        for row in event.hard:
            if abs(row[1]) in (21, 22, 23):
                rows.append(("hard", row))
        heavy = [p for p in event.particles if not p.final and _is_heavy_or_long(p, event)]
        if heavy:
            rows.append(("head", "DECAYED STATES"))
            rows += [("particle", p) for p in heavy]
        finals = sorted(event.finals(), key=lambda p: -p.p4.e)
        rows.append(("head", f"FINAL STATE · {len(finals)} particles by energy"))
        rows += [("particle", p) for p in finals[:300]]
        previous = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-4, -4))
        y = rect.y + 8 - self.scroll_tree
        for kind, item in rows:
            height = 26 if kind == "head" else 40
            if rect.y - 44 < y < rect.bottom + 4:
                if kind == "head":
                    self._blit("caption", str(item), ACCENT, (rect.x + 12, y + 8))
                elif kind == "hard":
                    pdg, status = int(item[0]), int(item[1])
                    role = {21: "incoming", 22: "intermediate", 23: "outgoing"}.get(abs(status), "")
                    energy = format_energy(float(item[7]))
                    pygame.draw.circle(self.screen, particle_color(pdg), (rect.x + 16, y + 9), 4)
                    name = self._blit("body", english_name(pdg), TEXT, (rect.x + 28, y))
                    self._blit("small", symbol(pdg), TEXT_3, (name.right + 8, y + 1))
                    self._blit("mono_small", f"{energy} · {role}", TEXT_3, (rect.x + 28, y + 21))
                else:
                    particle = item
                    pygame.draw.circle(self.screen, particle_color(particle.pdg), (rect.x + 16, y + 9), 4, 0 if particle.final else 1)
                    name = self._blit("body", self.text.fit("body", english_name(particle.pdg), rect.w - 110), TEXT if particle.final else TEXT_2, (rect.x + 28, y))
                    self._blit("small", symbol(particle.pdg), TEXT_3, (name.right + 8, y + 1))
                    detail = brief_result(particle).replace("   ", " · ")
                    self._blit("mono_small", self.text.fit("mono_small", detail, rect.w - 44), TEXT_3, (rect.x + 28, y + 21))
            y += height
        total = y + self.scroll_tree - rect.y
        self.scroll_tree = min(self.scroll_tree, max(0, total - rect.h + 8))
        self.screen.set_clip(previous)

    def _py_tab_charts(self, rect: pygame.Rect, draw_histogram) -> None:
        gap = 10
        top = pygame.Rect(rect.x, rect.y, rect.w, (rect.h - gap) // 2)
        bottom = pygame.Rect(rect.x, top.bottom + gap, rect.w, rect.h - top.h - gap)
        self.zone_hist = top
        self.zone_shape = bottom
        draw_histogram(self.screen, top, _auto_hist(self.py_shat, 30), "Hard scale √ŝ", self.fonts["ui_small"], self.fonts["mono_small"])
        draw_histogram(self.screen, bottom, _auto_hist(self.py_mult, 30), "Charged multiplicity",
                       self.fonts["ui_small"], self.fonts["mono_small"], unit="")
        self._hit(top, "help-hist")
        self._hit(bottom, "help-mult")
        clear = pygame.Rect(top.right - 82, top.y + 8, 72, 24)
        self._button(clear, "Clear", "clear", kind="ghost")

    # -- guide -------------------------------------------------------------------

    def _py_focus(self, action, payload) -> GuideCard | None:
        if action == "pypileup":
            return GuideCard(
                "Pileup",
                "At the LHC many proton pairs collide in the same bunch crossing. Delphes overlays a Poisson number of "
                "minimum-bias collisions with this mean, spread along the beam line as in the CMS card, and flags their "
                "tracks, which are drawn in grey from their own vertices. Charged pileup is removed from jets and missing "
                "momentum; neutral pileup energy remains, so the missing-momentum resolution degrades as μ grows. "
                "Run 3 averages about 60; the High-Luminosity LHC is designed for 140 to 200.",
            )
        if action in ("engine", "pybeam", "pyprocess", "pyslider", "pypreset"):
            key = {"pyslider": "pyenergy", "pypreset": "pyenergy"}.get(action, action)
            title, body = PY_HELP[key]
            return GuideCard(title, body)
        if action == "pytoggle":
            title, body = PY_HELP[f"pytoggle:{payload}"]
            return GuideCard(title, body)
        if action == "help-mult":
            return GuideCard(
                "Charged multiplicity",
                "The number of charged final-state particles in each event, all transverse momenta. "
                "In proton collisions most of them come from the underlying event and hadronization, not the hard process.",
            )
        if action == "help-hist" and self.pythia_mode:
            return GuideCard(
                "Hard scale √ŝ",
                "The invariant mass of the hard partonic collision in each generated event. In proton collisions it is "
                "set by the parton distributions; with lepton beams it falls below √s when photons are radiated.",
            )
        return None

    def _py_story(self) -> GuideCard:
        event = self.event if isinstance(self.event, PythiaEvent) else None
        if event is None:
            if self.pythia is not None and self.pythia.state == "error":
                return GuideCard("PYTHIA unavailable", self.pythia.message)
            return GuideCard(
                "PYTHIA 8.3",
                "Research-grade events: the hard process, QED and QCD radiation, multiparton interactions in proton "
                "collisions, Lund string hadronization, and hadron decays at their real decay points. "
                "Choose beams, energy, and a hard process, then press Collide.",
            )
        _approach, product = self._anim_phase()
        if product <= 0.0:
            return GuideCard("Approaching", "The two beams move toward the collision point. The motion is slowed so it can be followed.")
        counts = count_summary(event)
        met = math.hypot(self.report.missing_px, self.report.missing_py) if self.report else 0.0
        incoming, middle, outgoing = _hard_summary(event)
        parts = [
            f"{event.process_title} at √s = {format_energy(event.sqrt_s)}.",
            f"The hard collision ran at √ŝ = {format_energy(event.sqrt_s_hat)}: {' '.join(incoming)} → {' '.join(middle or outgoing)}.",
            f"PYTHIA then showered, hadronized, and decayed it into {len(event.finals())} final particles, "
            f"{counts['charged']} of them charged; {len(event.jets)} jets were found.",
            f"Charged tracks with pT above {MIN_PT_CHARGED:g} GeV are drawn from their production vertices; "
            f"{_displaced_vertices(event)} particles decayed visibly away from where they were made.",
        ]
        if met > 5.0:
            parts.append(f"Missing transverse momentum is {met:.1f} GeV.")
        if event.reco:
            parts.append(
                f"The detector response is simulated with {event.reco.get('detector', 'Delphes')}: the jets, leptons, "
                "photons, calorimeter towers, and missing momentum on screen are the reconstructed ones."
            )
        return GuideCard("This collision", " ".join(parts))


def _auto_hist(entries: list, bins: int) -> Histogram:
    """Histogram of (value, weight) pairs (or bare values) over their own range."""
    pairs = [entry if isinstance(entry, tuple) else (float(entry), 1.0) for entry in entries]
    if not pairs:
        return Histogram(0.0, 1.0, bins)
    lo = min(value for value, _w in pairs)
    hi = max(value for value, _w in pairs)
    if hi - lo < 1.0e-9:
        lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    hist = Histogram(max(0.0, lo - 0.05 * span), hi + 0.05 * span, bins)
    # LO events have unit weight; MC@NLO weights all have the same size, so their sign is exact.
    for value, weight in pairs:
        hist.fill(value, 1.0 if weight >= 0 else -1.0)
    return hist


def _hard_summary(event) -> tuple[list[str], list[str], list[str]]:
    incoming, middle, outgoing = [], [], []
    for row in event.hard:
        pdg, status = int(row[0]), abs(int(row[1]))
        if status == 21:
            incoming.append(symbol(pdg))
        elif status == 22:
            middle.append(symbol(pdg))
        elif status == 23:
            outgoing.append(symbol(pdg))
    return incoming, middle, outgoing


def _is_heavy_or_long(particle, event) -> bool:
    a = abs(particle.pdg)
    if a in (6, 23, 24, 25, 15):
        # PYTHIA keeps a copy after each recoil; the last copy is the one that decays.
        kids = event.daughters(particle)
        return bool(kids) and all(kid.pdg != particle.pdg for kid in kids)
    if abs(particle.status) < 81:
        return False
    decay = event.decay_vertex_mm(particle)
    return decay is not None and math.dist(decay, particle.vertex_mm) > 0.5


def _displaced_vertices(event) -> int:
    count = 0
    for particle in event.particles:
        if particle.final or abs(particle.status) < 81:
            continue
        decay = event.decay_vertex_mm(particle)
        if decay is not None and math.dist(decay, particle.vertex_mm) > 0.5:
            count += 1
    return count
