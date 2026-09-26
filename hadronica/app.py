"""Hadronica window: setup panel, event display, and results panel."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pygame

from hadronica import __version__
from hadronica.constants import B_SOLENOID_T, M_Z, PB_PER_GEV2, format_cross_section, format_energy
from hadronica.generator import PhysicsError, conservation_report, generate_event, isr_table, sigma_pb
from hadronica.guide import GuideCard, GuideContext, collision_card, focus_card, theta_degrees
from hadronica.app_pythia import PythiaMode
from hadronica.engine import PythiaEvent
from hadronica.fullscene import draw_full_event
from hadronica.histogram import Histogram
from hadronica.incoming import (
    COLLIDER_GROUPS,
    beam_for_initial,
    beta_speed,
    embed_in_lab,
    equal_headon_momentum,
    format_beta,
    incoming_momenta,
    invariant_sqrt_s,
)
from hadronica.methods import METHODS
from hadronica.particles import species
from hadronica.processes import BEAMS, all_processes, process_by_id
from hadronica.report import (
    brief_result,
    energy_timeline,
    english_name,
    generated_summary,
    motion_text,
    plain_outcome,
    report_cards,
    symbol,
)
from hadronica.report_sheet import draw_report_sheet
from hadronica.scene import (
    ECAL_COLOR,
    HCAL_COLOR,
    MUON_COLORS,
    SOLENOID_COLOR,
    TRACKER_COLOR,
    Camera,
    draw_event,
    draw_histogram,
    draw_incoming,
    draw_lineshape,
    draw_momentum_inset,
    draw_static_detector,
    paint_background,
)
from hadronica.theme import (
    ACCENT,
    ACCENT_HOVER,
    ACCENT_INK,
    ACCENT_SOFT,
    BAD,
    BG,
    BORDER,
    BORDER_STRONG,
    GOLD,
    GOOD,
    SURFACE,
    SURFACE_2,
    SURFACE_3,
    TEXT,
    TEXT_2,
    TEXT_3,
    WARN,
    TextCache,
    card,
    fill_round,
    load_fonts,
    particle_color,
    shadow,
)
from hadronica.tracks import curvature_radius
from hadronica.validation_view import ValidationRun, draw_validation, load_results
from hadronica.view3d import Orbit, draw_full_event_3d, draw_view3d

# The approach is a viewing aid. Physical times are computed separately.
APPROACH_SECONDS = 2.2
PRODUCT_PAUSE = 0.18
PRODUCT_SECONDS = 2.05

RANGES = {
    "low": (0.4, 30.0),
    "Z": (60.0, 140.0),
    "high": (150.0, 500.0),
}
PRESETS = (
    ("10 GeV", 10.0, "low"),
    ("Z pole", M_Z, "Z"),
    ("250 GeV", 250.0, "high"),
    ("500 GeV", 500.0, "high"),
)
TABS = (("summary", "Summary"), ("products", "Products"), ("charts", "Charts"))

TOPBAR_H = 60
STATUS_H = 30
GAP = 12
PAD = 16
BATCH_SIZE = 200


class LabApp(PythiaMode):
    def __init__(self, size: tuple[int, int] = (1480, 900), *, headless: bool = False, seed: int = 20240921):
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("Hadronica — Standard Model Collision Laboratory")
        flags = pygame.HIDDEN if headless else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_icon(self._icon())
        self.clock = pygame.time.Clock()
        self.fonts = load_fonts()
        self.text = TextCache(self.fonts)
        self.headless = headless
        self.base_seed = seed
        self.counter = 0
        # What collides.
        self.beam_id = "ee"
        self.custom = False
        self.picker: str | None = None
        self.pdg_a = 11
        self.pdg_b = -11
        self.momentum_a = equal_headon_momentum(species(11).mass, M_Z)
        self.momentum_b = self.momentum_a
        self.collide_angle = 180.0
        self.custom_supported = True
        self.a_is_plus = True
        self.lab_a = None
        self.lab_b = None
        self.lab_plus = None
        self.lab_minus = None
        self.process_id = "ff13"
        self.sqrt_s = M_Z
        self.range_name = "Z"
        # Physics and display options.
        self.isr = True
        self.force_muon = False
        self.b_field = B_SOLENOID_T
        self.pt_guide = False
        self.view_radius = 7.4
        self.view_3d = False
        self.orbit = Orbit()
        # Event state.
        self.event = None
        self.report = None
        self.error = ""
        self.anim = 0.0
        self.mass_hist = Histogram(0.0, 110.0, 48)
        self.batch_left = 0
        self.batch_total = 0
        self.rows: list[tuple] = []
        self.curve = None
        self.shown_sigma = 0.0
        self.born_sigma = 0.0
        # Interface state.
        self.tab = "summary"
        self._reveal_selected = True
        self._curves: dict[tuple, tuple] = {}
        self._physics_dirty = False
        self._last_refresh_ms = 0
        self._pythia_init_state()
        self.scroll_left = 0
        self.scroll_proc = 0
        self.scroll_tree = 0
        self.scroll_methods = 0
        self.scroll_guide = 0
        self._guide_key: tuple[str, str] | None = None
        self.show_methods = False
        self.show_report = False
        self.show_validation = False
        self.validation_data = None
        self.validation_path = ""
        self.validation_selected = None
        self.validation_plot = 0
        self.validation_run = ValidationRun()
        self._validation_was_running = False
        self.energy_focus = False
        self.energy_buffer = ""
        self.drag: str | None = None
        self.pointer_pos: tuple[int, int] | None = None
        self._held_focus = None
        self.guide_focus_body = ""
        self.hot: list[tuple[pygame.Rect, str, object]] = []
        self._clip: pygame.Rect | None = None
        self.zone_left = pygame.Rect(0, 0, 0, 0)
        self.zone_guide = pygame.Rect(0, 0, 0, 0)
        self.zone_hist = pygame.Rect(0, 0, 0, 0)
        self.zone_shape = pygame.Rect(0, 0, 0, 0)
        self.zone_inset = pygame.Rect(0, 0, 0, 0)
        self.zone_detector = pygame.Rect(0, 0, 0, 0)
        self.zone_process = pygame.Rect(0, 0, 0, 0)
        self.zone_tree = pygame.Rect(0, 0, 0, 0)
        self._static_key = None
        self._static = None
        self._bg = None
        self._bg_size = (0, 0)
        self.refresh_physics()
        self.spawn()

    # ------------------------------------------------------------------
    # Physics state
    # ------------------------------------------------------------------

    def _icon(self) -> pygame.Surface:
        """The window icon (hadronica/assets/window_icon.png, drawn by tools/make_icon.py)."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "window_icon.png")
        try:
            return pygame.image.load(path)
        except (pygame.error, FileNotFoundError):
            icon = pygame.Surface((64, 64), pygame.SRCALPHA)  # fallback: the same motif, drawn directly
            pygame.draw.rect(icon, (10, 13, 21), icon.get_rect(), border_radius=14)
            pygame.draw.circle(icon, (140, 116, 70), (32, 32), 23, 3)
            pygame.draw.circle(icon, (255, 255, 255), (32, 32), 6)
            return icon

    def refresh_physics(self) -> None:
        self._physics_dirty = False
        self._last_refresh_ms = pygame.time.get_ticks()
        beam = BEAMS[self.beam_id]
        use_isr = self.isr and beam.id in ("ee", "mumu")
        rows = []
        for process in all_processes():
            if process.allowed(beam, self.sqrt_s):
                rows.append((process, sigma_pb(process, beam, self.sqrt_s, use_isr)))
        if not any(process.id == self.process_id for process, _sigma in rows):
            rows.sort(key=lambda item: item[1], reverse=True)
            self.process_id = rows[0][0].id if rows else "ff13"
        rows.sort(key=lambda item: -item[1])
        self.rows = rows
        self._reveal_selected = True
        selected = process_by_id(self.process_id)
        self.shown_sigma = next((sigma for process, sigma in rows if process.id == self.process_id), 0.0)
        self.born_sigma = sigma_pb(selected, beam, self.sqrt_s, False) if rows else 0.0
        self.curve = self._lineshape()

    def _lineshape(self):
        process = process_by_id(self.process_id)
        beam = BEAMS[self.beam_id]
        if process.id == "zh":
            lo, hi = 210.0, 500.0
        elif process.id == "ff6":
            lo, hi = 340.0, 500.0
        elif self.sqrt_s < 40.0:
            lo, hi = 0.5, 30.0
        elif self.sqrt_s < 160.0:
            lo, hi = 50.0, 150.0
        else:
            lo, hi = 150.0, 500.0
        use_isr = self.isr and beam.id in ("ee", "mumu")
        key = (process.id, beam.id, use_isr, lo, hi)
        cached = self._curves.get(key)
        if cached is not None:
            return cached
        xs = np.linspace(lo, hi, 64)
        born = np.zeros_like(xs)
        radiated = np.zeros_like(xs)
        for i, energy in enumerate(xs):
            born[i] = sigma_pb(process, beam, float(energy), False)
            if use_isr and process.allowed(beam, float(energy)):
                radiated[i] = isr_table(process, beam, float(energy)).total * PB_PER_GEV2
            else:
                radiated[i] = born[i]
        self._curves[key] = (xs, born, radiated)
        return xs, born, radiated

    def _reset_histogram(self) -> None:
        hi = max(20.0, self.sqrt_s * 1.15)
        self.mass_hist = Histogram(0.0, hi, 48)

    def spawn(self, *, animate: bool = True) -> None:
        if self.pythia_mode:
            self._py_collide()
            return
        self._refresh_if_dirty(force=True)
        self.counter += 1
        seed = (self.base_seed + self.counter) & 0x7FFFFFFF
        if self.custom and not self.custom_supported:
            self.event = None
            self.report = None
            return
        try:
            event = generate_event(
                process_by_id(self.process_id),
                self.beam_id,
                self.sqrt_s,
                seed,
                isr=self.isr,
                force_muon=self.force_muon,
            )
            if self.custom and self.lab_plus is not None and self.lab_minus is not None:
                event = embed_in_lab(event, self.lab_a, self.lab_b, self.a_is_plus)
        except (PhysicsError, RuntimeError, ValueError) as exc:
            self.error = str(exc)
            self.event = None
            self.report = None
            return
        self.event = event
        self.report = conservation_report(event)
        self.error = ""
        self.anim = 0.0 if animate else 4.0
        self.scroll_tree = 0
        self.mass_hist.fill(event.sqrt_s_hat)

    def set_energy(self, value: float, *, from_slider: bool = False) -> None:
        value = min(500.0, max(0.4, value))
        changed = abs(value - self.sqrt_s) > 1.0e-6
        self.sqrt_s = value
        if not from_slider:
            if value < 40.0:
                self.range_name = "low"
            elif value < 145.0:
                self.range_name = "Z"
            else:
                self.range_name = "high"
        if changed:
            window = self.mass_hist.hi
            if self.sqrt_s > window or window > max(20.0, self.sqrt_s * 1.35):
                self._reset_histogram()
            if from_slider:
                # Folding every channel with the radiator takes ~0.1 s; while the
                # slider moves, draw() refreshes at most a few times per second.
                self._physics_dirty = True
            else:
                self.refresh_physics()

    def _refresh_if_dirty(self, *, force: bool = False) -> None:
        if not self._physics_dirty:
            return
        now = pygame.time.get_ticks()
        if force or self.drag != "energy" or now - self._last_refresh_ms > 200:
            self.refresh_physics()

    def _enter_custom(self) -> None:
        beam = BEAMS[self.beam_id]
        self.pdg_a = beam.pdg_plus
        self.pdg_b = beam.pdg_minus
        self.collide_angle = 180.0
        momentum = equal_headon_momentum(species(self.pdg_a).mass, self.sqrt_s)
        self.momentum_a = momentum
        self.momentum_b = momentum
        self.custom = True
        self.picker = None
        self._apply_custom()
        if self.custom_supported:
            self.spawn()

    def _apply_custom(self) -> None:
        self.lab_a, self.lab_b = incoming_momenta(
            self.pdg_a, self.momentum_a, self.pdg_b, self.momentum_b, self.collide_angle
        )
        sqrt_s = invariant_sqrt_s(self.lab_a, self.lab_b)
        self.sqrt_s = sqrt_s
        matched = beam_for_initial(self.pdg_a, self.pdg_b)
        if sqrt_s < 0.4 or matched is None:
            self.custom_supported = False
            self.a_is_plus = True
            self.lab_plus = None
            self.lab_minus = None
            self.rows = []
            self.curve = None
            self.shown_sigma = 0.0
            self.born_sigma = 0.0
            if sqrt_s < 0.4:
                self.error = "√s is under 0.4 GeV. Raise a momentum or open the angle toward 180°."
            else:
                self.error = "No Born formula for this incoming pair. The rays are the whole collision."
            self.event = None
            self.report = None
            return
        beam, self.a_is_plus = matched
        self.custom_supported = True
        self.lab_plus = self.lab_a if self.a_is_plus else self.lab_b
        self.lab_minus = self.lab_b if self.a_is_plus else self.lab_a
        self.beam_id = beam.id
        self.error = ""
        self.refresh_physics()

    # ------------------------------------------------------------------
    # Main loop and input
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._shutdown_engine()
                    pygame.quit()
                    return
                self._handle(event)
            self._advance_batch()
            self.anim += dt
            self.draw()
            pygame.display.flip()

    def _open_validation(self) -> None:
        self.validation_data, self.validation_path = load_results()
        self.show_validation = True

    def _shutdown_engine(self) -> None:
        if self.pythia is not None:
            self.pythia.stop()

    def _advance_batch(self) -> None:
        if self.batch_left <= 0 or self.pythia_mode:
            return
        for _ in range(4):
            if self.batch_left <= 0:
                break
            self.batch_left -= 1
            self.spawn(animate=self.batch_left == 0)

    def _start_batch(self) -> None:
        if self.pythia_mode:
            self._py_batch()
            return
        self.batch_total = BATCH_SIZE
        self.batch_left = BATCH_SIZE

    def _handle(self, event) -> None:
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(1180, event.w), max(760, event.h)), pygame.RESIZABLE)
            return
        if event.type == pygame.MOUSEWHEEL:
            self._wheel(pygame.mouse.get_pos(), event.y)
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._click(event.pos)
            return
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragged = self.drag
            self.drag = None
            if dragged in ("mom-a", "mom-b", "angle") and self.custom and self.custom_supported:
                self.spawn()
            return
        if event.type == pygame.MOUSEMOTION and self.drag:
            self._drag(event.pos)
            return
        if event.type == pygame.KEYDOWN:
            self._key(event)

    def _wheel(self, pos, dy: int) -> None:
        if self.show_methods:
            self.scroll_methods = max(0, self.scroll_methods - dy * 36)
        elif self.show_report or self.picker:
            return
        elif self.zone_process.collidepoint(pos):
            self.scroll_proc = max(0, self.scroll_proc - dy * 30)
        elif self.zone_tree.collidepoint(pos):
            self.scroll_tree = max(0, self.scroll_tree - dy * 30)
        elif self.zone_guide.collidepoint(pos):
            self.scroll_guide = max(0, self.scroll_guide - dy * 24)
        elif self.zone_left.collidepoint(pos):
            self.scroll_left = max(0, self.scroll_left - dy * 36)
        elif self.zone_detector.collidepoint(pos):
            factor = 0.9 if dy > 0 else 1.1
            if self.view_3d:
                self.orbit.distance = min(48.0, max(8.0, self.orbit.distance * factor))
            else:
                self.view_radius = min(12.0, max(1.15, self.view_radius * factor))

    def _key(self, event) -> None:
        if self.energy_focus:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._commit_energy()
            elif event.key == pygame.K_ESCAPE:
                self.energy_focus = False
            elif event.key == pygame.K_BACKSPACE:
                self.energy_buffer = self.energy_buffer[:-1]
            elif event.unicode and (event.unicode.isdigit() or event.unicode == ".") and len(self.energy_buffer) < 9:
                self.energy_buffer += event.unicode
            return
        if event.key == pygame.K_ESCAPE:
            if self.show_validation:
                self.show_validation = False
            elif self.show_report:
                self.show_report = False
            elif self.picker:
                self.picker = None
            elif self.show_methods:
                self.show_methods = False
            else:
                self._shutdown_engine()
                pygame.quit()
                sys.exit(0)
        elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
            self.spawn()
        elif event.key == pygame.K_b:
            self._start_batch()
        elif event.key == pygame.K_c:
            self._reset_histogram()
            self.py_shat.clear()
            self.py_mult.clear()
        elif event.key == pygame.K_v:
            self.view_3d = not self.view_3d
        elif event.key == pygame.K_r:
            if self.event is not None:
                self.show_report = True
        elif event.key == pygame.K_F1:
            self.show_methods = not self.show_methods
            self.scroll_methods = 0
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS) and self.pythia_mode:
            self._py_set_energy(self.py_energy[self.py_beam] * 1.01)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) and self.pythia_mode:
            self._py_set_energy(self.py_energy[self.py_beam] / 1.01)
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS) and not self.custom:
            self.set_energy(self.sqrt_s + (0.05 if self.range_name == "Z" else 0.5))
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) and not self.custom:
            self.set_energy(self.sqrt_s - (0.05 if self.range_name == "Z" else 0.5))

    def _commit_energy(self) -> None:
        self.energy_focus = False
        try:
            value = float(self.energy_buffer)
        except ValueError:
            return
        if self.pythia_mode:
            self._py_set_energy(value)
        else:
            self.set_energy(value)

    def _click(self, pos) -> None:
        if self.picker:
            for rect, action, payload in reversed(self.hot):
                if rect.collidepoint(pos) and action == "species":
                    _slot, pdg = payload
                    if self.picker == "a":
                        self.pdg_a = int(pdg)
                    else:
                        self.pdg_b = int(pdg)
                    self.picker = None
                    self._apply_custom()
                    if self.custom_supported:
                        self.spawn()
                    return
                if rect.collidepoint(pos) and action in ("close-picker", "picker-panel"):
                    if action == "close-picker":
                        self.picker = None
                    return
            self.picker = None
            return
        if self.show_report:
            for rect, action, _payload in reversed(self.hot):
                if action == "close-report" and rect.collidepoint(pos):
                    self.show_report = False
            return
        if self.show_methods:
            for rect, action, _payload in reversed(self.hot):
                if action == "close-methods" and rect.collidepoint(pos):
                    self.show_methods = False
            return
        if self.show_validation:
            for rect, action, payload in reversed(self.hot):
                if not rect.collidepoint(pos):
                    continue
                if action == "close-validation":
                    self.show_validation = False
                elif action == "val-select":
                    self.validation_selected = payload
                    self.validation_plot = 0
                elif action == "val-plot":
                    self.validation_plot += int(payload)
                elif action == "val-run":
                    self.validation_run.start()
                return
            return
        for rect, action, payload in reversed(self.hot):
            if not rect.collidepoint(pos):
                continue
            if action.startswith("help-"):
                continue
            self._activate(action, payload, pos)
            return
        self.energy_focus = False
        if self.view_3d and self.zone_detector.collidepoint(pos):
            self.drag = "orbit"
            self._orbit_last = pos

    def _activate(self, action: str, payload, pos) -> None:
        if action != "energy":
            self.energy_focus = False
        if self._py_activate(action, payload, pos):
            return
        if action == "view":
            self.view_3d = payload == "3d"
        elif action == "view3d":
            self.view_3d = not self.view_3d
        elif action == "tab":
            self.tab = str(payload)
        elif action == "beam":
            if payload == "custom":
                self._enter_custom()
                return
            self.custom = False
            self.picker = None
            self.error = ""
            self.beam_id = str(payload)
            self.scroll_proc = 0
            self._reset_histogram()
            self.refresh_physics()
        elif action == "open-picker":
            self.picker = str(payload)
        elif action in ("mom-a", "mom-b", "angle"):
            self.drag = action
            self._drag(pos)
        elif action == "range":
            self.range_name = str(payload)
            lo, hi = RANGES[self.range_name]
            if not lo <= self.sqrt_s <= hi:
                self.set_energy(0.5 * (lo + hi), from_slider=True)
        elif action == "preset":
            _label, energy, range_name = payload
            self.range_name = range_name
            self.set_energy(energy)
        elif action == "process":
            self.process_id = str(payload)
            self.refresh_physics()
        elif action == "toggle":
            setattr(self, str(payload), not getattr(self, str(payload)))
            if payload == "isr":
                self.refresh_physics()
        elif action == "collide":
            self.spawn()
        elif action == "batch":
            self._start_batch()
        elif action == "clear":
            self._reset_histogram()
            self.py_shat.clear()
            self.py_mult.clear()
        elif action == "methods":
            self.show_methods = True
            self.scroll_methods = 0
        elif action == "validation":
            self._open_validation()
        elif action == "report":
            if self.event is not None:
                self.show_report = True
        elif action == "energy":
            self.energy_focus = True
            value = self.py_energy[self.py_beam] if self.pythia_mode else self.sqrt_s
            self.energy_buffer = f"{value:.3f}" if value < 1000 else f"{value:.1f}"
        elif action == "slider":
            self.drag = "energy"
            self._drag(pos)
        elif action == "bslider":
            self.drag = "field"
            self._drag(pos)

    def _drag(self, pos) -> None:
        if self.drag == "energy" and hasattr(self, "_energy_rect"):
            lo, hi = RANGES[self.range_name]
            t = (pos[0] - self._energy_rect.x) / max(1, self._energy_rect.w)
            self.set_energy(lo + min(1.0, max(0.0, t)) * (hi - lo), from_slider=True)
        elif self.drag == "pyenergy" and hasattr(self, "_energy_rect"):
            self._py_drag_energy(pos)
        elif self.drag == "field" and hasattr(self, "_field_rect"):
            t = (pos[0] - self._field_rect.x) / max(1, self._field_rect.w)
            self.b_field = min(1.0, max(0.0, t)) * 4.0
        elif self.drag in ("mom-a", "mom-b", "angle"):
            rect = getattr(self, f"_{self.drag.replace('-', '_')}_rect", None)
            if rect is None:
                return
            t = min(1.0, max(0.0, (pos[0] - rect.x) / max(1, rect.w)))
            if self.drag == "angle":
                self.collide_angle = t * 180.0
            elif self.drag == "mom-a":
                self.momentum_a = t * 250.0
            else:
                self.momentum_b = t * 250.0
            self._apply_custom()
        elif self.drag == "orbit":
            dx = pos[0] - self._orbit_last[0]
            dy = pos[1] - self._orbit_last[1]
            self.orbit.yaw += dx * 0.008
            self.orbit.pitch = max(-1.15, min(1.15, self.orbit.pitch - dy * 0.008))
            self._orbit_last = pos

    def _zone(self, name: str) -> pygame.Rect:
        return getattr(self, f"zone_{name}", pygame.Rect(0, 0, 0, 0))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _mouse(self) -> tuple[int, int]:
        if self.pointer_pos is not None:
            return self.pointer_pos
        if self.headless:
            return (-1, -1)
        return pygame.mouse.get_pos()

    def _hovered(self, rect: pygame.Rect) -> bool:
        if self.picker or self.show_report or self.show_methods:
            return False
        pos = self._mouse()
        if self._clip is not None and not self._clip.collidepoint(pos):
            return False
        return rect.collidepoint(pos)

    def _hit(self, rect: pygame.Rect, action: str, payload=None) -> None:
        if self._clip is not None:
            rect = rect.clip(self._clip)
        if rect.w > 0 and rect.h > 0:
            self.hot.append((rect, action, payload))

    def draw(self) -> None:
        self._py_poll()
        self._refresh_if_dirty()
        self.hot = []
        self._clip = None
        width, height = self.screen.get_size()
        if self._bg_size != (width, height):
            self._bg = paint_background((width, height))
            self._bg_size = (width, height)
            self._static_key = None
        self.screen.blit(self._bg, (0, 0))

        body_y = TOPBAR_H + GAP
        body_h = height - TOPBAR_H - STATUS_H - 2 * GAP
        left_w = int(min(360, max(304, width * 0.235)))
        right_w = int(min(400, max(318, width * 0.25)))
        left = pygame.Rect(GAP, body_y, left_w, body_h)
        right = pygame.Rect(width - GAP - right_w, body_y, right_w, body_h)
        stage = pygame.Rect(left.right + GAP, body_y, right.x - GAP - (left.right + GAP), body_h)
        self.zone_detector = stage

        self._draw_topbar(pygame.Rect(0, 0, width, TOPBAR_H))
        self._draw_left(left)
        self._draw_stage(stage)
        self._draw_right(right)
        self._draw_status(pygame.Rect(0, height - STATUS_H, width, STATUS_H))
        if self.picker:
            self._draw_picker()
        if self.show_report and self.event is not None:
            self._draw_collision_report()
        if self.show_methods:
            self._draw_methods()
        if self.show_validation:
            if self._validation_was_running and not self.validation_run.running:
                self.validation_data, self.validation_path = load_results()
            self._validation_was_running = self.validation_run.running
            panel = self._modal_frame(1220, 780, "Validation against published data", "close-validation")
            draw_validation(self, panel)

    # ------------------------------------------------------------------
    # Small widgets
    # ------------------------------------------------------------------

    def _blit(self, key: str, text: str, color, pos, anchor: str = "topleft") -> pygame.Rect:
        image = self.text.render(key, text, color)
        rect = image.get_rect(**{anchor: pos})
        self.screen.blit(image, rect)
        return rect

    def _button(self, rect: pygame.Rect, label: str, action: str, payload=None, *, kind: str = "secondary") -> None:
        hover = self._hovered(rect)
        if kind == "primary":
            fill = ACCENT_HOVER if hover else ACCENT
            pygame.draw.rect(self.screen, fill, rect, border_radius=10)
            color = ACCENT_INK
            key = "h2"
        elif kind == "ghost":
            if hover:
                pygame.draw.rect(self.screen, SURFACE_3, rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)
            color = TEXT
            key = "small"
        else:
            pygame.draw.rect(self.screen, SURFACE_3 if hover else SURFACE_2, rect, border_radius=9)
            pygame.draw.rect(self.screen, BORDER_STRONG if hover else BORDER, rect, 1, border_radius=9)
            color = TEXT
            key = "body"
        shown = self.text.fit(key, label, rect.w - 16)
        self._blit(key, shown, color, rect.center, "center")
        self._hit(rect, action, payload)

    def _chip(self, rect: pygame.Rect, label: str, selected: bool, action: str, payload) -> None:
        hover = self._hovered(rect)
        if selected:
            pygame.draw.rect(self.screen, ACCENT_SOFT, rect, border_radius=rect.h // 2)
            pygame.draw.rect(self.screen, ACCENT, rect, 1, border_radius=rect.h // 2)
            color = TEXT
        else:
            pygame.draw.rect(self.screen, SURFACE_3 if hover else SURFACE_2, rect, border_radius=rect.h // 2)
            pygame.draw.rect(self.screen, BORDER_STRONG if hover else BORDER, rect, 1, border_radius=rect.h // 2)
            color = TEXT_2 if not hover else TEXT
        self._blit("small", label, color, rect.center, "center")
        self._hit(rect, action, payload)

    def _chips(self, x: int, y: int, width: int, items, action: str, height: int = 28) -> int:
        cursor = x
        for label, payload, selected in items:
            w = self.text.size("small", label)[0] + 24
            if cursor + w > x + width and cursor > x:
                cursor = x
                y += height + 8
            self._chip(pygame.Rect(cursor, y, w, height), label, selected, action, payload)
            cursor += w + 8
        return y + height

    def _segmented(self, rect: pygame.Rect, items, action: str) -> None:
        pygame.draw.rect(self.screen, SURFACE_2, rect, border_radius=9)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=9)
        count = len(items)
        seg_w = rect.w / count
        for index, (label, payload, selected) in enumerate(items):
            seg = pygame.Rect(int(rect.x + index * seg_w) + 3, rect.y + 3, int(seg_w) - 6, rect.h - 6)
            if selected:
                pygame.draw.rect(self.screen, ACCENT_SOFT, seg, border_radius=7)
                pygame.draw.rect(self.screen, ACCENT, seg, 1, border_radius=7)
            elif self._hovered(seg):
                pygame.draw.rect(self.screen, SURFACE_3, seg, border_radius=7)
            color = TEXT if selected else TEXT_2
            self._blit("small", self.text.fit("small", label, seg.w - 8), color, seg.center, "center")
            self._hit(seg, action, payload)

    def _slider(self, rect: pygame.Rect, t: float, action: str) -> None:
        t = min(1.0, max(0.0, t))
        track = pygame.Rect(rect.x, rect.centery - 3, rect.w, 6)
        pygame.draw.rect(self.screen, SURFACE_3, track, border_radius=3)
        fill = track.copy()
        fill.w = max(6, int(track.w * t))
        pygame.draw.rect(self.screen, ACCENT, fill, border_radius=3)
        knob = (rect.x + int(rect.w * t), rect.centery)
        hover = self._hovered(rect.inflate(0, 12)) or self.drag == action
        pygame.draw.circle(self.screen, ACCENT_INK, knob, 10 if hover else 9)
        pygame.draw.circle(self.screen, TEXT, knob, 8 if hover else 7)
        self._hit(rect.inflate(0, 14), action)

    def _switch(self, rect: pygame.Rect, label: str, value: bool, field: str) -> None:
        hover = self._hovered(rect)
        if hover:
            pygame.draw.rect(self.screen, SURFACE_2, rect, border_radius=8)
        self._blit("body", label, TEXT if hover else TEXT_2, (rect.x + 8, rect.centery), "midleft")
        track = pygame.Rect(rect.right - 46, rect.centery - 11, 38, 22)
        pygame.draw.rect(self.screen, ACCENT if value else SURFACE_3, track, border_radius=11)
        knob_x = track.right - 11 if value else track.x + 11
        pygame.draw.circle(self.screen, ACCENT_INK if value else TEXT_2, (knob_x, track.centery), 8)
        self._hit(rect, "toggle", field)

    def _step_header(self, x: int, y: int, width: int, number: str, title: str, detail: str = "") -> int:
        badge = pygame.Rect(x, y, 22, 22)
        pygame.draw.rect(self.screen, ACCENT_SOFT, badge, border_radius=11)
        self._blit("caption", number, ACCENT, badge.center, "center")
        title_rect = self._blit("h2", title, TEXT, (x + 30, y + 11), "midleft")
        if detail:
            room = x + width - title_rect.right - 12
            shown = self.text.fit("small", detail, room)
            self._blit("small", shown, TEXT_3, (x + width, y + 11), "midright")
        return y + 32

    def _wrap_block(self, x: int, y: int, width: int, text: str, key: str = "small", color=TEXT_2, line_h: int = 18) -> int:
        for line in self.text.wrap(key, text, width):
            self._blit(key, line, color, (x, y))
            y += line_h
        return y

    # ------------------------------------------------------------------
    # Top bar and status bar
    # ------------------------------------------------------------------

    def _draw_topbar(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, SURFACE, rect)
        pygame.draw.line(self.screen, BORDER, (0, rect.bottom - 1), (rect.w, rect.bottom - 1))
        cx, cy = 34, rect.centery
        pygame.draw.circle(self.screen, ACCENT, (cx, cy), 15, 2)
        pygame.draw.circle(self.screen, (44, 70, 120), (cx, cy), 8, 2)
        pygame.draw.line(self.screen, (88, 214, 255), (cx, cy), (cx + 11, cy - 8), 2)
        pygame.draw.line(self.screen, GOLD, (cx, cy), (cx - 9, cy + 9), 2)
        brand = self._blit("brand", "Hadronica", TEXT, (60, cy - 1), "midleft")
        subtitle = self._blit("small", "Standard Model Collision Laboratory", TEXT_2, (brand.right + 12, cy), "midleft")
        self._engine_switch(subtitle.right + 28, cy - 16, 250)

        x = rect.right - GAP
        methods = pygame.Rect(0, cy - 17, 170, 34)
        methods.right = x
        self._button(methods, "Methods & sources", "methods", kind="ghost")
        validation = pygame.Rect(0, cy - 17, 112, 34)
        validation.right = methods.x - 8
        self._button(validation, "Validation", "validation", kind="ghost")
        report = pygame.Rect(0, cy - 17, 120, 34)
        report.right = validation.x - 8
        self._button(report, "Full report", "report", kind="ghost")
        badge_text = "PDG 2026 · CODATA 2022 inputs"
        if self.pythia_mode:
            version = self.pythia.version if self.pythia is not None and self.pythia.version else "8.3"
            badge_text = f"PYTHIA {version} · Monash 2013 tune · PDG 2026 masses"
        badge_w = self.text.size("caption", badge_text)[0] + 22
        badge = pygame.Rect(0, cy - 12, badge_w, 24)
        badge.right = report.x - 14
        if badge.x > brand.right + 560:
            pygame.draw.rect(self.screen, SURFACE_2, badge, border_radius=12)
            pygame.draw.rect(self.screen, BORDER, badge, 1, border_radius=12)
            self._blit("caption", badge_text, TEXT_2, badge.center, "center")
            self._hit(badge, "methods")

    def _draw_status(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, SURFACE, rect)
        pygame.draw.line(self.screen, BORDER, (0, rect.y), (rect.w, rect.y))
        hints = (
            ("Space", "collide"),
            ("B", f"run {BATCH_SIZE}"),
            ("C", "clear histogram"),
            ("V", "2D / 3D"),
            ("+ −", "energy"),
            ("R", "report"),
            ("F1", "methods"),
        )
        x = 14
        for key, text in hints:
            key_image = self.text.render("caption", key, TEXT)
            box = pygame.Rect(x, rect.centery - 9, key_image.get_width() + 12, 18)
            pygame.draw.rect(self.screen, SURFACE_3, box, border_radius=5)
            self.screen.blit(key_image, key_image.get_rect(center=box.center))
            label = self._blit("small", text, TEXT_3, (box.right + 6, rect.centery), "midleft")
            x = label.right + 16
        right = f"Hadronica {__version__}"
        if self.event is not None:
            count = self.py_event_count if self.pythia_mode else self.counter
            right = f"event {count}   ·   seed {self.event.seed}   ·   " + right
        if self.batch_left:
            right = f"running {self.batch_total - self.batch_left}/{self.batch_total}   ·   " + right
        self._blit("small", right, TEXT_3, (rect.right - 14, rect.centery), "midright")

    # ------------------------------------------------------------------
    # Left panel: setup
    # ------------------------------------------------------------------

    def _draw_left(self, rect: pygame.Rect) -> None:
        card(self.screen, rect, fill=SURFACE)
        footer_h = 74
        footer = pygame.Rect(rect.x, rect.bottom - footer_h, rect.w, footer_h)
        region = pygame.Rect(rect.x + 1, rect.y + 1, rect.w - 2, rect.h - footer_h - 1)
        self.zone_left = region
        x = rect.x + PAD
        w = rect.w - 2 * PAD

        previous = self.screen.get_clip()
        self.screen.set_clip(region)
        self._clip = region
        y = region.y + 14 - self.scroll_left
        remaining_after = 206
        if self.pythia_mode:
            y = self._py_setup_particles(x, y, w)
            y = self._divider(x, y + 12, w)
            y = self._py_setup_energy(x, y, w)
            y = self._divider(x, y + 12, w)
            y = self._py_setup_final_state(x, y, w, region.bottom - remaining_after)
            y = self._divider(x, y + 12, w)
            y = self._py_setup_options(x, y, w)
        else:
            y = self._setup_particles(x, y, w)
            y = self._divider(x, y + 12, w)
            y = self._setup_energy(x, y, w)
            y = self._divider(x, y + 12, w)
            y = self._setup_final_state(x, y, w, region.bottom - remaining_after)
            y = self._divider(x, y + 12, w)
            y = self._setup_options(x, y, w)
        content_bottom = y + 14 + self.scroll_left
        self.scroll_left = min(self.scroll_left, max(0, content_bottom - region.bottom))
        self._clip = None
        self.screen.set_clip(previous)

        if content_bottom - region.bottom > 0:
            # Scroll indicator.
            span = region.h
            total = content_bottom - region.y
            bar_h = max(28, int(span * span / total))
            bar_y = region.y + int((span - bar_h) * self.scroll_left / max(1, total - span))
            pygame.draw.rect(self.screen, SURFACE_3, pygame.Rect(rect.right - 6, bar_y, 3, bar_h), border_radius=2)
            fade = pygame.Surface((region.w, 18), pygame.SRCALPHA)
            for i in range(18):
                pygame.draw.line(fade, (*SURFACE, int(200 * i / 18)), (0, i), (region.w, i))
            self.screen.blit(fade, (region.x, region.bottom - 18))

        pygame.draw.line(self.screen, BORDER, (footer.x + 1, footer.y), (footer.right - 2, footer.y))
        batch_w = 112
        collide = pygame.Rect(x, footer.y + 14, w - batch_w - 8, 46)
        ready = self.pythia_mode or not (self.custom and not self.custom_supported)
        label = "Collide" if ready else "No products"
        if self.pythia_mode and self.py_expect:
            label = "Generating…"
        self._button(collide, label, "collide", kind="primary" if ready else "secondary")
        batch_label = (
            f"{self.batch_total - self.batch_left}/{self.batch_total}"
            if self.batch_left
            else f"Run {BATCH_SIZE}"
        )
        self._button(pygame.Rect(collide.right + 8, collide.y, batch_w, 46), batch_label, "batch")

    def _divider(self, x: int, y: int, w: int) -> int:
        pygame.draw.line(self.screen, BORDER, (x, y), (x + w, y))
        return y + 12

    def _pair_caption(self) -> str:
        if self.custom:
            return f"{english_name(self.pdg_a)} and {english_name(self.pdg_b)}"
        beam = BEAMS[self.beam_id]
        return f"{english_name(beam.pdg_plus)} and {english_name(beam.pdg_minus)}"

    def _outcome_caption(self) -> str:
        process = process_by_id(self.process_id)
        return plain_outcome(process.id, process.title)

    def _setup_particles(self, x: int, y: int, w: int) -> int:
        beam = BEAMS[self.beam_id]
        if self.custom:
            detail = "your own pair and angle"
        elif beam.id in ("ee", "mumu"):
            detail = "lepton collider"
        else:
            detail = "single partons, not protons"
        y = self._step_header(x, y, w, "1", "Colliding particles", detail)
        items = [(mode.label, mode.id, mode.id == self.beam_id and not self.custom) for mode in BEAMS.values()]
        items.append(("Custom…", "custom", self.custom))
        y = self._chips(x, y, w, items, "beam")
        if self.custom:
            y = self._setup_custom(x, y + 14, w)
        return y

    def _setup_custom(self, x: int, y: int, w: int) -> int:
        half = (w - 8) // 2
        self._particle_button(pygame.Rect(x, y, half, 40), self.pdg_a, "a")
        self._particle_button(pygame.Rect(x + half + 8, y, w - half - 8, 40), self.pdg_b, "b")
        y += 52
        y = self._param_slider(
            x, y, w, "Momentum A", f"{self.momentum_a:.1f} GeV · {format_beta(beta_speed(self.pdg_a, self.momentum_a))}",
            self.momentum_a / 250.0, "mom-a",
        )
        y = self._param_slider(
            x, y, w, "Momentum B", f"{self.momentum_b:.1f} GeV · {format_beta(beta_speed(self.pdg_b, self.momentum_b))}",
            self.momentum_b / 250.0, "mom-b",
        )
        tone = "head-on" if self.collide_angle >= 175 else ("same direction" if self.collide_angle <= 5 else "crossing")
        y = self._param_slider(
            x, y, w, "Angle between momenta", f"{self.collide_angle:.0f}° · {tone}", self.collide_angle / 180.0, "angle"
        )
        box = pygame.Rect(x, y, w, 36)
        color = ACCENT if self.custom_supported else WARN
        pygame.draw.rect(self.screen, SURFACE_2, box, border_radius=9)
        self._blit("small", "√s from these 4-vectors", TEXT_2, (box.x + 12, box.centery), "midleft")
        self._blit("mono", f"{self.sqrt_s:.3f} GeV", color, (box.right - 12, box.centery), "midright")
        return box.bottom

    def _particle_button(self, rect: pygame.Rect, pdg: int, slot: str) -> None:
        hover = self._hovered(rect)
        active = self.picker == slot
        pygame.draw.rect(self.screen, SURFACE_3 if hover else SURFACE_2, rect, border_radius=9)
        pygame.draw.rect(self.screen, ACCENT if active else (BORDER_STRONG if hover else BORDER), rect, 1, border_radius=9)
        pygame.draw.circle(self.screen, particle_color(pdg), (rect.x + 16, rect.centery), 6)
        self._blit("caption", slot.upper(), TEXT_3, (rect.x + 30, rect.y + 5))
        name = self.text.fit("body", english_name(pdg), rect.w - 40)
        self._blit("body", name, TEXT, (rect.x + 30, rect.y + 17))
        self._hit(rect, "open-picker", slot)

    def _param_slider(self, x: int, y: int, w: int, label: str, value: str, t: float, action: str) -> int:
        self._blit("small", label, TEXT_2, (x, y))
        self._blit("mono_small", value, TEXT, (x + w, y + 2), "topright")
        rect = pygame.Rect(x + 4, y + 22, w - 8, 18)
        setattr(self, f"_{action.replace('-', '_')}_rect", rect)
        self._slider(rect, t, action)
        return y + 50

    def _setup_energy(self, x: int, y: int, w: int) -> int:
        y = self._step_header(x, y, w, "2", "Collision energy", "√s, center of mass")
        if self.custom:
            return self._wrap_block(
                x, y, w, "Set by the two momenta and the angle above.", "small", TEXT_3
            )
        box = pygame.Rect(x, y, w, 48)
        hover = self._hovered(box)
        pygame.draw.rect(self.screen, SURFACE_3 if hover else SURFACE_2, box, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT if self.energy_focus else (BORDER_STRONG if hover else BORDER), box, 1, border_radius=10)
        shown = self.energy_buffer if self.energy_focus else f"{self.sqrt_s:.3f}"
        number = self._blit("big", shown, TEXT, (box.x + 14, box.centery), "midleft")
        if self.energy_focus and (int(self.anim * 2) % 2 == 0 or self.headless):
            pygame.draw.line(self.screen, ACCENT, (number.right + 2, box.y + 12), (number.right + 2, box.bottom - 12), 2)
        self._blit("body", "GeV", TEXT_2, (number.right + 10, box.centery + 3), "midleft")
        hint = "type, then Enter" if self.energy_focus else "click to type"
        self._blit("caption", hint, TEXT_3, (box.right - 12, box.centery), "midright")
        self._hit(box, "energy")
        y = box.bottom + 16
        lo, hi = RANGES[self.range_name]
        self._energy_rect = pygame.Rect(x + 4, y, w - 8, 18)
        self._slider(self._energy_rect, (self.sqrt_s - lo) / (hi - lo), "slider")
        y += 22
        self._blit("mono_small", f"{lo:g}", TEXT_3, (x, y))
        self._blit("mono_small", f"{hi:g} GeV", TEXT_3, (x + w, y), "topright")
        y += 20
        return self._chips(
            x,
            y,
            w,
            [(label, (label, energy, name), abs(self.sqrt_s - energy) < 0.02) for label, energy, name in PRESETS],
            "preset",
            height=26,
        )

    def _setup_final_state(self, x: int, y: int, w: int, soft_bottom: int) -> int:
        beam = BEAMS[self.beam_id]
        use_isr = self.isr and beam.id in ("ee", "mumu")
        detail = "σ with ISR" if use_isr else "Born σ"
        y = self._step_header(x, y, w, "3", "Final state", detail)
        if self.custom and not self.custom_supported:
            return self._wrap_block(x, y, w, "This pair has no Born formula, so there is no final state to choose.", "small", TEXT_3)
        row_h = 34
        full = len(self.rows) * row_h + 8
        list_h = int(max(136, min(full, soft_bottom - y)))
        if self._reveal_selected:
            self._reveal_selected = False
            index = next((i for i, (process, _s) in enumerate(self.rows) if process.id == self.process_id), 0)
            row_top = index * row_h
            if row_top < self.scroll_proc or row_top + row_h > self.scroll_proc + list_h - 8:
                self.scroll_proc = max(0, row_top - (list_h - row_h) // 2)
        rect = pygame.Rect(x, y, w, list_h)
        self.zone_process = rect.clip(self._clip) if self._clip is not None else rect
        pygame.draw.rect(self.screen, BG, rect, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=10)
        previous_clip = self._clip
        inner = rect.inflate(-4, -4)
        view = inner.clip(previous_clip) if previous_clip is not None else inner
        saved = self.screen.get_clip()
        self.screen.set_clip(view)
        self._clip = view
        self.scroll_proc = min(self.scroll_proc, max(0, full - list_h))
        row_y = rect.y + 4 - self.scroll_proc
        top = max(self.rows[0][1], 1e-300) if self.rows else 1.0
        for process, sigma in self.rows:
            row = pygame.Rect(rect.x + 4, row_y, rect.w - 8, row_h - 2)
            selected = process.id == self.process_id
            hover = self._hovered(row)
            if selected:
                pygame.draw.rect(self.screen, ACCENT_SOFT, row, border_radius=7)
                pygame.draw.rect(self.screen, ACCENT, pygame.Rect(row.x, row.y + 6, 3, row.h - 12), border_radius=2)
            elif hover:
                pygame.draw.rect(self.screen, SURFACE_2, row, border_radius=7)
            value = self.text.render("mono_small", format_cross_section(sigma), TEXT if selected else TEXT_2)
            label_room = row.w - value.get_width() - 28
            label = self.text.fit("body", plain_outcome(process.id, process.title), label_room)
            self._blit("body", label, TEXT if (selected or hover) else TEXT_2, (row.x + 12, row.centery - 1), "midleft")
            self.screen.blit(value, value.get_rect(midright=(row.right - 10, row.centery - 3)))
            # Share of the largest rate, on a log scale over twelve decades.
            share = 0.0 if sigma <= 0 else max(0.0, 1.0 + math.log10(max(sigma, 1e-300) / top) / 12.0)
            bar = pygame.Rect(row.right - 10 - 60, row.bottom - 7, int(60 * share), 2)
            if bar.w > 0:
                pygame.draw.rect(self.screen, ACCENT if selected else BORDER_STRONG, bar)
            self._hit(row, "process", process.id)
            row_y += row_h
        self._clip = previous_clip
        self.screen.set_clip(saved)
        if full > list_h:
            span = list_h - 8
            bar_h = max(20, int(span * list_h / full))
            bar_y = rect.y + 4 + int((span - bar_h) * self.scroll_proc / max(1, full - list_h))
            pygame.draw.rect(self.screen, SURFACE_3, pygame.Rect(rect.right - 5, bar_y, 3, bar_h), border_radius=2)
        return rect.bottom

    def _setup_options(self, x: int, y: int, w: int) -> int:
        y = self._step_header(x, y, w, "4", "Physics & detector")
        options = (
            ("Initial-state radiation", self.isr, "isr"),
            ("Muons decay (cτ = 659 m)", self.force_muon, "force_muon"),
            ("pT reference curves", self.pt_guide, "pt_guide"),
        )
        for label, value, field in options:
            self._switch(pygame.Rect(x - 8, y, w + 16, 32), label, value, field)
            y += 33
        y += 6
        self._blit("small", "Solenoid field", TEXT_2, (x, y))
        self._blit("mono_small", f"{self.b_field:.2f} T", TEXT, (x + w, y + 2), "topright")
        self._field_rect = pygame.Rect(x + 4, y + 22, w - 8, 18)
        self._slider(self._field_rect, self.b_field / 4.0, "bslider")
        return y + 44

    # ------------------------------------------------------------------
    # Center: detector stage
    # ------------------------------------------------------------------

    def _draw_stage(self, rect: pygame.Rect) -> None:
        card(self.screen, rect, fill=(8, 11, 19))
        self.zone_inset = pygame.Rect(0, 0, 0, 0)
        # Registered first so every widget drawn on the stage takes precedence.
        self._hit(rect, "help-detector")
        inner = rect.inflate(-2, -2)
        previous = self.screen.get_clip()
        self.screen.set_clip(inner)
        if self.view_3d:
            self._draw_detector_3d(inner)
        else:
            self._draw_detector_2d(inner)
        self.screen.set_clip(previous)
        self._stage_overlay(rect)

    def _draw_detector_2d(self, rect: pygame.Rect) -> None:
        camera = Camera(rect, self.view_radius)
        key = (rect.x, rect.y, rect.w, rect.h, round(self.view_radius, 3))
        if key != self._static_key:
            plate = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            local = Camera(pygame.Rect(0, 0, rect.w, rect.h), self.view_radius)
            draw_static_detector(plate, local, self.b_field)
            self._static = plate
            self._static_key = key
        self.screen.blit(self._static, rect.topleft)
        approach, product = self._anim_phase()
        pair = self._py_approach_pair() if self.pythia_mode else self._approach_pair()
        if pair is not None and (self.event is None or product <= 0.0):
            momentum_a, momentum_b, pdg_a, pdg_b, angle = pair
            draw_incoming(
                self.screen,
                camera,
                momentum_a,
                momentum_b,
                pdg_a,
                pdg_b,
                angle,
                self.fonts["label"],
                progress=0.0 if self.event is None else approach,
            )
        if isinstance(self.event, PythiaEvent) and product > 0.0:
            draw_full_event(
                self.screen,
                camera,
                self.event,
                self._py_traces_for(self.event),
                product,
                self.fonts["label"],
                self.report,
            )
            if rect.width > 520 and rect.height > 420:
                inset = pygame.Rect(rect.x + 14, rect.bottom - 150, 236, 136)
                self.zone_inset = inset
                draw_momentum_inset(self.screen, inset, self.event, product, self.fonts["mono_small"])
                self._hit(inset, "help-inset")
        elif self.event is not None and product > 0.0:
            draw_event(
                self.screen,
                camera,
                self.event,
                fraction=product,
                b_field=self.b_field,
                show_guide=self.pt_guide,
                font=self.fonts["label"],
                report=self.report,
            )
            if rect.width > 520 and rect.height > 420:
                inset = pygame.Rect(rect.x + 14, rect.bottom - 150, 236, 136)
                self.zone_inset = inset
                draw_momentum_inset(self.screen, inset, self.event, product, self.fonts["mono_small"])
                self._hit(inset, "help-inset")
        if self.event is not None:
            self._flash(camera)

    def _draw_detector_3d(self, rect: pygame.Rect) -> None:
        approach, product = self._anim_phase()
        start = APPROACH_SECONDS - 0.04
        flash = 0.0
        if start < self.anim < start + 0.62:
            flash = (1.0 - (self.anim - start) / 0.62) ** 2
        if isinstance(self.event, PythiaEvent) and product > 0.0:
            draw_full_event_3d(
                self.screen, rect, self.orbit, self.event, self._py_traces_for(self.event),
                product, self.fonts["label"], flash,
            )
            return
        draw_view3d(
            self.screen,
            rect,
            self.orbit,
            self.event,
            approach,
            product,
            self.b_field,
            self.fonts["label"],
            flash,
        )

    def _stage_overlay(self, rect: pygame.Rect) -> None:
        # View switch, top right.
        switch = pygame.Rect(0, rect.y + 12, 196, 32)
        switch.right = rect.right - 12
        self._segmented(
            switch,
            [("Transverse 2D", "2d", not self.view_3d), ("3D barrel", "3d", self.view_3d)],
            "view",
        )
        # What is on screen, top left.
        x = rect.x + 14
        y = rect.y + 14
        _approach, product = self._anim_phase()
        if self.event is not None:
            title = self.event.process_title
            hard = "√ŝ" if isinstance(self.event, PythiaEvent) else "√s′"
            sub = f"{hard} = {format_energy(self.event.sqrt_s_hat)}   ·   B = {self.b_field:.2f} T"
        elif self.pythia_mode:
            title, sub = self._py_stage_title()
        elif self.custom and not self.custom_supported:
            title = f"{species(self.pdg_a).name} + {species(self.pdg_b).name}"
            sub = "Incoming rays only"
        else:
            title = BEAMS[self.beam_id].label
            sub = f"√s = {format_energy(self.sqrt_s)}   ·   B = {self.b_field:.2f} T"
        room = switch.x - x - 16
        title_img = self.text.render("h2", self.text.fit("h2", title, room - 20), TEXT)
        sub_img = self.text.render("small", self.text.fit("small", sub, room - 20), TEXT_2)
        pill = pygame.Rect(x, y, max(title_img.get_width(), sub_img.get_width()) + 24, 50)
        fill_round(self.screen, pill, (12, 16, 27, 215), 10)
        self.screen.blit(title_img, (pill.x + 12, pill.y + 7))
        self.screen.blit(sub_img, (pill.x + 12, pill.y + 28))
        if self.event is not None and product <= 0.0 and self.anim < APPROACH_SECONDS:
            self._blit("caption", "APPROACHING · SLOWED FOR VIEWING", ACCENT, (pill.x + 2, pill.bottom + 8))

        # Error banner.
        if self.error:
            text = self.text.fit("small", self.error, rect.w - 80)
            image = self.text.render("small", text, TEXT)
            banner = pygame.Rect(0, 0, image.get_width() + 40, 34)
            banner.midtop = (rect.centerx, rect.y + 70)
            fill_round(self.screen, banner, (70, 24, 34, 235), 10)
            pygame.draw.rect(self.screen, BAD, banner, 1, border_radius=10)
            pygame.draw.circle(self.screen, BAD, (banner.x + 16, banner.centery), 4)
            self.screen.blit(image, (banner.x + 28, banner.centery - image.get_height() // 2))

        # Hint and legend, bottom right.
        hint = "Drag to turn · scroll to zoom" if self.view_3d else "Scroll to zoom"
        hint_img = self.text.render("caption", hint, TEXT_3)
        self.screen.blit(hint_img, hint_img.get_rect(bottomright=(rect.right - 14, rect.bottom - 12)))
        if not self.view_3d and rect.h > 460:
            self._legend(rect.right - 14, rect.bottom - 34)

    def _legend(self, right: int, bottom: int) -> None:
        rows = (
            (TRACKER_COLOR, "Tracker"),
            (ECAL_COLOR, "ECAL · e, γ"),
            (HCAL_COLOR, "HCAL · quarks, g, π"),
            (SOLENOID_COLOR, "Solenoid"),
            (MUON_COLORS[0], "Muon stations"),
        )
        width = max(self.text.size("caption", label)[0] for _c, label in rows) + 34
        box = pygame.Rect(0, 0, width, len(rows) * 18 + 14)
        box.bottomright = (right, bottom)
        fill_round(self.screen, box, (12, 16, 27, 205), 10)
        y = box.y + 8
        for color, label in rows:
            pygame.draw.rect(self.screen, color, pygame.Rect(box.x + 10, y + 3, 12, 10), border_radius=3)
            self._blit("caption", label, TEXT_2, (box.x + 28, y + 1))
            y += 18

    def _anim_phase(self) -> tuple[float, float]:
        """Approach progress, then product progress. Both run from 0 to 1."""
        if self.anim < APPROACH_SECONDS:
            return self.anim / APPROACH_SECONDS, 0.0
        grown = self.anim - APPROACH_SECONDS - PRODUCT_PAUSE
        if grown <= 0.0:
            return 1.0, 0.0
        return 1.0, min(1.0, grown / PRODUCT_SECONDS)

    def _approach_pair(self):
        """Momenta drawn in the screen plane, aimed at the vertex."""
        if self.custom:
            if self.lab_a is None or self.lab_b is None:
                return None
            return self.lab_a, self.lab_b, self.pdg_a, self.pdg_b, self.collide_angle
        beam = BEAMS[self.beam_id]
        momentum = equal_headon_momentum(species(beam.pdg_plus).mass, self.sqrt_s)
        plus, minus = incoming_momenta(beam.pdg_plus, momentum, beam.pdg_minus, momentum, 180.0)
        return plus, minus, beam.pdg_plus, beam.pdg_minus, 180.0

    def _flash(self, camera: Camera) -> None:
        start = APPROACH_SECONDS - 0.04
        if not (start < self.anim < start + 0.62):
            return
        u = (self.anim - start) / 0.62
        radius = int(18 + u * min(camera.rect.w, camera.rect.h) * 0.22)
        alpha = int(180 * (1.0 - u) ** 2)
        layer = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(layer, (210, 230, 255, alpha), (radius + 2, radius + 2), radius)
        pygame.draw.circle(layer, (255, 255, 255, min(255, alpha + 40)), (radius + 2, radius + 2), max(4, radius // 5))
        cx, cy = camera.origin
        self.screen.blit(layer, (cx - radius - 2, cy - radius - 2))

    # ------------------------------------------------------------------
    # Right panel: results
    # ------------------------------------------------------------------

    def _draw_right(self, rect: pygame.Rect) -> None:
        card(self.screen, rect, fill=SURFACE)
        x = rect.x + PAD
        w = rect.w - 2 * PAD
        y = self._conservation_card(x, rect.y + PAD, w)
        tabs = pygame.Rect(x, y + 12, w, 34)
        self._segmented(tabs, [(label, key, self.tab == key) for key, label in TABS], "tab")
        guide_h = 184 if rect.h >= 700 else 150
        guide = pygame.Rect(rect.x + 1, rect.bottom - guide_h, rect.w - 2, guide_h - 1)
        content = pygame.Rect(x, tabs.bottom + 14, w, guide.y - tabs.bottom - 26)
        self.zone_tree = pygame.Rect(0, 0, 0, 0)
        self.zone_hist = pygame.Rect(0, 0, 0, 0)
        self.zone_shape = pygame.Rect(0, 0, 0, 0)
        if self.pythia_mode:
            if self.tab == "products":
                self._py_tab_products(content)
            elif self.tab == "charts":
                self._py_tab_charts(content, draw_histogram)
            else:
                self._py_tab_summary(content)
        elif self.tab == "products":
            self._tab_products(content)
        elif self.tab == "charts":
            self._tab_charts(content)
        else:
            self._tab_summary(content)
        self._draw_guide(guide)

    def _conservation_card(self, x: int, y: int, w: int) -> int:
        box = pygame.Rect(x, y, w, 58)
        if self.report is None:
            pygame.draw.rect(self.screen, SURFACE_2, box, border_radius=10)
            message = self.error or "No event yet. Press Collide."
            color = WARN if self.error else TEXT_2
            lines = self.text.wrap("small", message, w - 24)[:2]
            ly = box.centery - 9 * len(lines)
            for line in lines:
                self._blit("small", line, color, (box.x + 12, ly))
                ly += 18
            return box.bottom
        ok = self.report.ok
        tone = GOOD if ok else BAD
        fill_round(self.screen, box, (*tone, 22), 10)
        pygame.draw.rect(self.screen, tuple(int(c * 0.55) for c in tone), box, 1, border_radius=10)
        icon = (box.x + 22, box.centery)
        pygame.draw.circle(self.screen, tone, icon, 11)
        if ok:
            pygame.draw.lines(self.screen, ACCENT_INK, False, [(icon[0] - 5, icon[1]), (icon[0] - 1, icon[1] + 4), (icon[0] + 5, icon[1] - 4)], 3)
        else:
            pygame.draw.line(self.screen, ACCENT_INK, (icon[0] - 4, icon[1] - 4), (icon[0] + 4, icon[1] + 4), 3)
            pygame.draw.line(self.screen, ACCENT_INK, (icon[0] + 4, icon[1] - 4), (icon[0] - 4, icon[1] + 4), 3)
        headline = "Conservation laws hold" if ok else "Conservation FAILED"
        self._blit("h2", headline, TEXT, (box.x + 44, box.y + 9))
        charge = self.report.delta_charge_thirds
        charge_text = str(charge // 3) if charge % 3 == 0 else f"{charge}/3"
        lepton = ",".join(str(v) for v in self.report.delta_lepton)
        detail = (
            f"ΔE {abs(self.report.delta_e):.0e} Δp {self.report.delta_p:.0e} "
            f"ΔQ {charge_text} ΔB {self.report.delta_baryon_thirds} ΔL {lepton}"
        )
        if getattr(self.report, "checks_flavor", True) is False:
            detail = f"ΔE {abs(self.report.delta_e):.0e} GeV   Δp {self.report.delta_p:.0e} GeV   ΔQ {charge_text}"
        self._blit("mono_small", self.text.fit("mono_small", detail, w - 54), TEXT_2, (box.x + 44, box.y + 33))
        self._hit(box, "help-conservation")
        return box.bottom

    def _tab_summary(self, rect: pygame.Rect) -> None:
        tile_w = (rect.w - 8) // 2
        tile_h = 62
        beam = BEAMS[self.beam_id]
        use_isr = self.isr and beam.id in ("ee", "mumu")
        _approach, product = self._anim_phase()
        revealed = self.event is not None and product > 0.0
        if revealed:
            hard = format_energy(self.event.sqrt_s_hat)
            hard_sub = "full beam energy" if self.event.isr_v <= 1.0e-4 else f"of {format_energy(self.event.sqrt_s)} after ISR"
            angle = f"{theta_degrees(self.event.cos_theta):.1f}°"
            met = math.hypot(self.report.missing_px, self.report.missing_py) if self.report else 0.0
            met_text = f"{met:.2f} GeV"
        elif self.event is not None:
            hard, hard_sub, angle, met_text = "…", "colliding", "…", "…"
        else:
            hard, hard_sub, angle, met_text = "—", "no event yet", "—", "—"
        tiles = (
            ("Hard scale √s′", hard, hard_sub),
            ("Cross section", format_cross_section(self.shown_sigma), f"Born {format_cross_section(self.born_sigma)}" if use_isr else "Born, no radiator"),
            ("Primary angle θ", angle, "from the incoming particle"),
            ("Missing pT", met_text, "from neutrinos"),
        )
        for index, (label, value, sub) in enumerate(tiles):
            col, row = index % 2, index // 2
            tile = pygame.Rect(rect.x + col * (tile_w + 8), rect.y + row * (tile_h + 8), tile_w, tile_h)
            pygame.draw.rect(self.screen, SURFACE_2, tile, border_radius=10)
            self._blit("caption", label.upper(), TEXT_3, (tile.x + 12, tile.y + 9))
            self._blit("h2", self.text.fit("h2", value, tile.w - 24), TEXT, (tile.x + 12, tile.y + 24))
            self._blit("caption", self.text.fit("caption", sub, tile.w - 24), TEXT_3, (tile.x + 12, tile.y + 45))
            help_action = {0: "help-hist", 1: "help-shape", 2: "help-detector", 3: "help-detector"}[index]
            self._hit(tile, help_action)
        y = rect.y + 2 * tile_h + 8 + 16
        if self.event is None:
            text = (
                "Pick the particles, the energy, and the final state on the left, then press Collide. "
                "This panel will list what was produced, its energy, and how long each product lives."
            )
            if self.custom and not self.custom_supported:
                text = self.error or text
            self._wrap_block(rect.x, y, rect.w, text, "small", TEXT_2)
            return
        if not revealed:
            self._wrap_block(
                rect.x, y, rect.w,
                "The two particles are approaching. What they produce appears here when they meet.",
                "small", TEXT_2,
            )
            return
        self._blit("caption", "WHAT WAS PRODUCED", TEXT_3, (rect.x, y))
        link = pygame.Rect(0, y - 5, 118, 24)
        link.right = rect.right
        self._button(link, "Full report", "report", kind="ghost")
        y += 26
        previous = self.screen.get_clip()
        area = pygame.Rect(rect.x, y, rect.w, rect.bottom - y)
        self.screen.set_clip(area)
        for title, lines in report_cards(self.event):
            if title == "Energy":
                continue
            body = " · ".join(lines)
            wrapped = self.text.wrap("small", body, rect.w - 24)
            box = pygame.Rect(rect.x, y, rect.w, 30 + 17 * len(wrapped))
            if box.y > area.bottom:
                break
            pygame.draw.rect(self.screen, SURFACE_2, box, border_radius=10)
            self._blit("body", self.text.fit("body", title[:1].upper() + title[1:], box.w - 24), TEXT, (box.x + 12, box.y + 7))
            ly = box.y + 27
            for line in wrapped:
                self._blit("small", line, TEXT_2, (box.x + 12, ly))
                ly += 17
            y = box.bottom + 8
        self.screen.set_clip(previous)

    def _tab_products(self, rect: pygame.Rect) -> None:
        if self.event is None:
            self._wrap_block(rect.x, rect.y, rect.w, "The full decay chain appears here after a collision.", "small", TEXT_2)
            return
        self.zone_tree = rect
        pygame.draw.rect(self.screen, BG, rect, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=10)
        self._hit(rect, "help-tree")
        previous = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-4, -4))
        y = rect.y + 10 - self.scroll_tree
        self._blit("mono_small", f"seed {self.event.seed}  ·  lab-frame energies", TEXT_3, (rect.x + 12, y))
        y += 24
        for index, particle in enumerate(self.event.particles):
            if particle.status == "beam":
                continue
            depth = _depth(self.event.particles, index)
            decayed = particle.status == "intermediate"
            shown = "photon, radiated (ISR)" if particle.status == "isr" else english_name(particle.pdg)
            symbol = "γ" if particle.status == "isr" else species(particle.pdg).name
            indent = rect.x + 12 + depth * 16
            if rect.y - 40 < y < rect.bottom + 4:
                if depth:
                    pygame.draw.line(self.screen, BORDER_STRONG, (indent - 10, y - 4), (indent - 10, y + 9))
                    pygame.draw.line(self.screen, BORDER_STRONG, (indent - 10, y + 9), (indent - 4, y + 9))
                pygame.draw.circle(self.screen, particle_color(particle.pdg), (indent + 4, y + 9), 4 if not decayed else 3, 0 if not decayed else 1)
                name = self._blit("body", f"{shown}", TEXT_2 if decayed else TEXT, (indent + 14, y))
                self._blit("small", symbol, TEXT_3, (name.right + 8, y + 1))
                detail = brief_result(particle).replace("   ", " · ")
                self._blit("mono_small", self.text.fit("mono_small", detail, rect.right - indent - 26), TEXT_3, (indent + 14, y + 22))
            y += 46
        total = y + self.scroll_tree - rect.y
        self.scroll_tree = min(self.scroll_tree, max(0, total - rect.h + 8))
        self.screen.set_clip(previous)

    def _tab_charts(self, rect: pygame.Rect) -> None:
        gap = 10
        top = pygame.Rect(rect.x, rect.y, rect.w, (rect.h - gap) // 2)
        bottom = pygame.Rect(rect.x, top.bottom + gap, rect.w, rect.h - top.h - gap)
        self.zone_hist = top
        self.zone_shape = bottom
        draw_histogram(self.screen, top, self.mass_hist, "Hard scale √s′", self.fonts["ui_small"], self.fonts["mono_small"])
        self._hit(top, "help-hist")
        clear = pygame.Rect(top.right - 82, top.y + 8, 72, 24)
        self._button(clear, "Clear", "clear", kind="ghost")
        if self.curve is not None:
            xs, born, radiated = self.curve
            draw_lineshape(
                self.screen,
                bottom,
                xs,
                born,
                radiated,
                self.sqrt_s,
                "Cross section vs √s",
                self.fonts["ui_small"],
                self.fonts["mono_small"],
            )
        self._hit(bottom, "help-shape")

    def _draw_guide(self, rect: pygame.Rect) -> None:
        self.zone_guide = rect
        pygame.draw.line(self.screen, BORDER, (rect.x + PAD, rect.y), (rect.right - PAD, rect.y))
        action, payload = self._current_focus()
        if self.pythia_mode:
            focus = self._py_focus(action, payload)
            if focus is None and action in ("collide", "batch", "clear", "tab", "view", "view3d", "report", "methods",
                                            "bslider", "help-inset", "help-tree", "help-conservation", "energy"):
                focus = focus_card(action, payload, self._guide_context())
            story = self._py_story()
            if focus is not None:
                shown, label = focus, "ABOUT THIS"
            else:
                shown, label = story, ("THIS COLLISION" if isinstance(self.event, PythiaEvent) else "GUIDE")
            self.guide_focus_body = shown.body
            self._guide_body(rect, shown, label)
            return
        context = self._guide_context()
        focus = focus_card(action, payload, context)
        self.guide_focus_body = focus.body
        if action not in (None, "help-detector"):
            shown = focus
            label = "ABOUT THIS"
        elif context.approaching:
            shown = collision_card(context)
            label = "WATCH"
        elif self.event is None and not self.custom:
            shown = GuideCard(
                "Start here",
                "Choose who collides, the energy, and what should come out, then press Collide. "
                "Point at any control, plot, or the detector and this card explains what it changes.",
            )
            label = "GUIDE"
        else:
            shown = collision_card(context)
            label = "THIS COLLISION"
        self._guide_body(rect, shown, label)

    def _guide_body(self, rect: pygame.Rect, shown, label: str) -> None:
        key = (shown.title, shown.body)
        if key != self._guide_key:
            self._guide_key = key
            self.scroll_guide = 0
        x = rect.x + PAD
        w = rect.w - 2 * PAD
        self._blit("caption", label, ACCENT, (x, rect.y + 12))
        self._blit("h2", self.text.fit("h2", shown.title, w), TEXT, (x, rect.y + 28))
        body = pygame.Rect(x, rect.y + 52, w, rect.bottom - rect.y - 60)
        lines = self.text.wrap("small", shown.body, w - 8)
        total = len(lines) * 18
        self.scroll_guide = min(self.scroll_guide, max(0, total - body.h))
        previous = self.screen.get_clip()
        self.screen.set_clip(body)
        y = body.y - self.scroll_guide
        for line in lines:
            if body.y - 18 < y < body.bottom:
                self._blit("small", line, TEXT_2, (x, y))
            y += 18
        self.screen.set_clip(previous)
        if total > body.h:
            fade = pygame.Surface((w, 16), pygame.SRCALPHA)
            for i in range(16):
                pygame.draw.line(fade, (*SURFACE, int(230 * i / 16)), (0, i), (w, i))
            self.screen.blit(fade, (x, body.bottom - 16))
            more = "scroll for more" if self.scroll_guide < total - body.h else ""
            if more:
                self._blit("caption", more, TEXT_3, (rect.right - PAD, rect.y + 12), "topright")

    def _guide_context(self) -> GuideContext:
        sqrt_s_hat = None
        photon_energy = None
        theta = None
        finals: tuple[int, ...] = ()
        tight_name = None
        tight_pt = None
        tight_radius = None
        met = 0.0
        if self.event is not None:
            sqrt_s_hat = self.event.sqrt_s_hat
            theta = theta_degrees(self.event.cos_theta)
            photon = self.event.isr_photon()
            if photon is not None:
                photon_energy = photon.p4.e
            finals = tuple(particle.pdg for particle in self.event.finals())
            best_radius = None
            for particle in self.event.finals():
                charge = getattr(particle, "charge", None)
                if charge is None:
                    charge = species(particle.pdg).charge
                radius = curvature_radius(particle.p4.pt, charge, self.b_field)
                if radius is None:
                    continue
                if best_radius is None or radius < best_radius:
                    best_radius = radius
                    tight_name = symbol(particle.pdg)
                    tight_pt = particle.p4.pt
                    tight_radius = radius
        if self.report is not None:
            met = math.hypot(self.report.missing_px, self.report.missing_py)
        approach, product = self._anim_phase()
        approaching = self.event is not None and approach < 1.0
        generated = (
            generated_summary(self.event)
            if self.event is not None and product > 0.0 and not isinstance(self.event, PythiaEvent)
            else ""
        )
        if approaching and self.event is not None:
            if self.custom:
                name_a = species(self.pdg_a).name
                name_b = species(self.pdg_b).name
                angle = self.collide_angle
            elif isinstance(self.event, PythiaEvent):
                beams = self.event.beams()
                name_a = symbol(beams[0].pdg) if beams else "beam"
                name_b = symbol(beams[1].pdg) if len(beams) > 1 else "beam"
                angle = 180.0
            else:
                beam = BEAMS[self.beam_id]
                name_a = species(beam.pdg_plus).name
                name_b = species(beam.pdg_minus).name
                angle = 180.0
            scale = self.event.sqrt_s_hat or self.event.sqrt_s
            moving = motion_text(name_a, name_b, angle, scale, same_frame=self.custom)
        else:
            moving = ""
        return GuideContext(
            beam_id=self.beam_id,
            process_id=self.process_id,
            sqrt_s=self.sqrt_s,
            shown_pb=self.shown_sigma,
            born_pb=self.born_sigma,
            isr=self.isr,
            force_muon=self.force_muon,
            pt_guide=self.pt_guide,
            b_field=self.b_field,
            range_name=self.range_name,
            sqrt_s_hat=sqrt_s_hat,
            isr_photon_energy=photon_energy,
            theta_deg=theta,
            final_pdgs=finals,
            tight_name=tight_name,
            tight_pt=tight_pt,
            tight_radius_m=tight_radius,
            met=met,
            hist_entries=self.mass_hist.entries,
            custom=self.custom,
            custom_supported=self.custom_supported,
            pdg_a=self.pdg_a,
            pdg_b=self.pdg_b,
            momentum_a=self.momentum_a,
            momentum_b=self.momentum_b,
            angle_deg=self.collide_angle,
            beta_a=beta_speed(self.pdg_a, self.momentum_a),
            beta_b=beta_speed(self.pdg_b, self.momentum_b),
            approaching=approaching,
            motion_text=moving,
            generated_text=generated,
        )

    def _current_focus(self):
        pos = self.pointer_pos
        if pos is None and not self.headless:
            pos = pygame.mouse.get_pos()
        if pos is None:
            return None, None
        if self.zone_guide.collidepoint(pos):
            return self._held_focus if self._held_focus is not None else (None, None)
        for rect, action, payload in reversed(self.hot):
            if rect.w > 0 and rect.h > 0 and rect.collidepoint(pos):
                self._held_focus = (action, payload)
                return action, payload
        self._held_focus = None
        return None, None

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _modal_frame(self, width: int, height: int, title: str, close_action: str) -> pygame.Rect:
        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((4, 6, 12, 190))
        self.screen.blit(shade, (0, 0))
        panel = pygame.Rect(0, 0, min(width, self.screen.get_width() - 48), min(height, self.screen.get_height() - 48))
        panel.center = self.screen.get_rect().center
        shadow(self.screen, panel)
        card(self.screen, panel, fill=SURFACE, border=BORDER_STRONG, radius=18)
        self._blit("h1", title, TEXT, (panel.x + 24, panel.y + 18))
        close = pygame.Rect(panel.right - 52, panel.y + 16, 34, 34)
        pygame.draw.rect(self.screen, SURFACE_2, close, border_radius=9)
        pygame.draw.line(self.screen, TEXT, (close.x + 11, close.y + 11), (close.right - 11, close.bottom - 11), 2)
        pygame.draw.line(self.screen, TEXT, (close.right - 11, close.y + 11), (close.x + 11, close.bottom - 11), 2)
        self.hot.append((close, close_action, None))
        return panel

    def _draw_picker(self) -> None:
        which = "A" if self.picker == "a" else "B"
        panel = self._modal_frame(600, 600, f"Choose particle {which}", "close-picker")
        self.hot.insert(0, (panel, "picker-panel", None))
        y = panel.y + 58
        x0 = panel.x + 24
        self._wrap_block(
            x0, y, panel.w - 48,
            "Products are generated for e⁻e⁺, μ⁻μ⁺, or a quark with its own antiquark. Other pairs are drawn as incoming rays only.",
            "small", TEXT_3,
        )
        y += 44
        columns = 4
        col_w = (panel.w - 48) // columns
        current = self.pdg_a if self.picker == "a" else self.pdg_b
        pos = self._mouse()
        for group, pdgs in COLLIDER_GROUPS:
            self._blit("caption", group.upper(), TEXT_3, (x0, y))
            y += 20
            for index, pdg in enumerate(pdgs):
                col = index % columns
                row = index // columns
                rect = pygame.Rect(x0 + col * col_w, y + row * 38, col_w - 8, 32)
                selected = pdg == current
                hover = rect.collidepoint(pos)
                pygame.draw.rect(self.screen, ACCENT_SOFT if selected else (SURFACE_3 if hover else SURFACE_2), rect, border_radius=8)
                pygame.draw.rect(self.screen, ACCENT if selected else BORDER, rect, 1, border_radius=8)
                pygame.draw.circle(self.screen, particle_color(pdg), (rect.x + 14, rect.centery), 5)
                self._blit("body", species(pdg).name, TEXT, (rect.x + 26, rect.centery), "midleft")
                self.hot.append((rect, "species", (self.picker, pdg)))
            rows = (len(pdgs) + columns - 1) // columns
            y += rows * 38 + 10

    def _draw_collision_report(self) -> None:
        meet = APPROACH_SECONDS
        end = APPROACH_SECONDS + PRODUCT_PAUSE + PRODUCT_SECONDS
        timeline = energy_timeline(self.event, meet, end)
        close = draw_report_sheet(self.screen, self.fonts, self.event, timeline)
        self.hot.append((close, "close-report", None))

    def _draw_methods(self) -> None:
        panel = self._modal_frame(820, 720, "Methods & sources", "close-methods")
        body = pygame.Rect(panel.x + 28, panel.y + 66, panel.w - 56, panel.h - 84)
        previous = self.screen.get_clip()
        self.screen.set_clip(body)
        y = body.y - self.scroll_methods
        for paragraph in METHODS.split("\n"):
            text = paragraph.strip()
            if not text:
                y += 8
                continue
            if text.isupper() and len(text) < 40:
                y += 10
                self._blit("caption", text, ACCENT, (body.x, y))
                y += 22
                continue
            formula = any(token in text for token in ("χ(s) =", "σ = [", "dσ/dΩ =", "H(v) ="))
            key = "mono" if formula and len(text) < 110 else "body"
            color = TEXT if key == "body" else ACCENT_HOVER
            for line in self.text.wrap(key, text, body.w - 8):
                if body.y - 24 < y < body.bottom:
                    self._blit(key, line, color, (body.x, y))
                y += 21
            y += 4
        total = y + self.scroll_methods - body.y
        self.scroll_methods = min(self.scroll_methods, max(0, total - body.h))
        self.screen.set_clip(previous)
        if total > body.h:
            span = body.h
            bar_h = max(30, int(span * span / total))
            bar_y = body.y + int((span - bar_h) * self.scroll_methods / max(1, total - span))
            pygame.draw.rect(self.screen, SURFACE_3, pygame.Rect(panel.right - 12, bar_y, 4, bar_h), border_radius=2)


def _depth(particles, index: int) -> int:
    depth = 0
    parent = particles[index].parent
    seen = set()
    while parent is not None and parent not in seen:
        seen.add(parent)
        depth += 1
        parent = particles[parent].parent
    return depth


def main() -> None:
    args = sys.argv[1:]
    if "--selftest" in args:
        # Headless end-to-end check of this build (see hadronica/selftest.py); writes a JSON report.
        from hadronica.selftest import run_selftest

        after = args[args.index("--selftest") + 1:]
        report = after[0] if after and not after[0].startswith("--") else "hadronica-selftest.json"
        sys.exit(run_selftest(report))
    try:
        app = LabApp()
        if "--pythia" in sys.argv[1:]:
            app._activate("engine", "pythia", (0, 0))
        app.run()
    except SystemExit:
        raise
    except Exception as exc:
        _report_crash(exc)
        raise


def _report_crash(exc: BaseException) -> None:
    import traceback

    text = traceback.format_exc()
    try:
        path = "hadronica-error.log"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        path = ""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, f"{exc}\n\n{path}", "Hadronica", 0x10)
