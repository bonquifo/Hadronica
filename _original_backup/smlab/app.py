"""SMLab window: controls, event display, and the lineshape."""

from __future__ import annotations

import math
import sys

import numpy as np
import pygame

from smlab.constants import B_SOLENOID_T, M_Z, PB_PER_GEV2, format_cross_section, format_energy
from smlab.generator import PhysicsError, conservation_report, convolute_born, generate_event, sigma_pb
from smlab.guide import GuideContext, collision_card, focus_card, theta_degrees
from smlab.report import (
    brief_result,
    energy_timeline,
    english_name,
    generated_summary,
    motion_text,
    plain_outcome,
    report_cards,
)
from smlab.report_sheet import draw_report_sheet
from smlab.incoming import (
    COLLIDER_GROUPS,
    beam_for_initial,
    beta_speed,
    embed_in_lab,
    equal_headon_momentum,
    format_beta,
    incoming_momenta,
    invariant_sqrt_s,
)
from smlab.histogram import Histogram
from smlab.methods import METHODS
from smlab.particles import species
from smlab.processes import BEAMS, all_processes, process_by_id
from smlab.tracks import curvature_radius
from smlab.view3d import Orbit, draw_view3d

# The approach is a viewing aid. Physical times are computed separately.
APPROACH_SECONDS = 2.2
PRODUCT_PAUSE = 0.18
PRODUCT_SECONDS = 2.05
from smlab.scene import (
    Camera,
    draw_event,
    draw_histogram,
    draw_incoming,
    draw_lineshape,
    draw_momentum_inset,
    draw_static_detector,
    make_radial_glow,
    paint_background,
)
from smlab.theme import (
    BAD,
    BG_RAISED,
    CYAN,
    DIM,
    GOLD,
    GOOD,
    HAIRLINE,
    MUTED,
    PANEL,
    PANEL_EDGE,
    TEXT,
    WARN,
    load_fonts,
)

RANGES = {
    "low": (0.4, 30.0),
    "Z": (60.0, 140.0),
    "high": (150.0, 500.0),
}
PRESETS = (
    ("10", 10.0, "low"),
    ("Z", M_Z, "Z"),
    ("250", 250.0, "high"),
    ("500", 500.0, "high"),
)


class LabApp:
    def __init__(self, size: tuple[int, int] = (1480, 900), *, headless: bool = False, seed: int = 20240921):
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("SMLab — Standard Model Collision Laboratory")
        flags = pygame.HIDDEN if headless else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_icon(self._icon())
        self.clock = pygame.time.Clock()
        self.fonts = load_fonts()
        self.headless = headless
        self.base_seed = seed
        self.counter = 0
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
        self.isr = True
        self.force_muon = False
        self.b_field = B_SOLENOID_T
        self.pt_guide = False
        self.view_radius = 7.4
        self.view_3d = False
        self.orbit = Orbit()
        self.event = None
        self.report = None
        self.error = ""
        self.anim = 0.0
        self.mass_hist = Histogram(0.0, 110.0, 48)
        self.scroll_proc = 0
        self.scroll_tree = 0
        self.scroll_methods = 0
        self.show_methods = False
        self.show_report = False
        self._report_for = None
        self.more_open = False
        self.menu: str | None = None
        self.sections = {
            "particles": False,
            "energy": False,
            "products": False,
            "detector": False,
            "charts": False,
            "event": False,
        }
        self.energy_focus = False
        self.energy_buffer = ""
        self.drag: str | None = None
        self.batch_left = 0
        self.batch_total = 0
        self.rows: list[tuple] = []
        self.curve = None
        self.shown_sigma = 0.0
        self.born_sigma = 0.0
        self.pointer_pos: tuple[int, int] | None = None
        self._held_focus = None
        self.guide_focus_body = ""
        self.zone_guide = pygame.Rect(0, 0, 0, 0)
        self.zone_hist = pygame.Rect(0, 0, 0, 0)
        self.zone_shape = pygame.Rect(0, 0, 0, 0)
        self.zone_inset = pygame.Rect(0, 0, 0, 0)
        self.zone_detector = pygame.Rect(0, 0, 0, 0)
        self._static_key = None
        self._static = None
        self._bg = None
        self._bg_size = (0, 0)
        self._halo = make_radial_glow(560, (40, 90, 150), 36)
        self.hot: list[tuple[pygame.Rect, str, object]] = []
        self.refresh_physics()
        self.spawn()

    def _icon(self) -> pygame.Surface:
        icon = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.circle(icon, (16, 28, 48), (32, 32), 30)
        pygame.draw.circle(icon, GOLD, (32, 32), 30, 2)
        pygame.draw.circle(icon, (18, 46, 82), (32, 32), 16, 3)
        pygame.draw.line(icon, CYAN, (32, 32), (54, 18), 3)
        pygame.draw.line(icon, (132, 164, 255), (32, 32), (14, 48), 3)
        return icon

    def refresh_physics(self) -> None:
        beam = BEAMS[self.beam_id]
        rows = []
        for process in all_processes():
            if process.allowed(beam, self.sqrt_s):
                rows.append((process, sigma_pb(process, beam, self.sqrt_s, self.isr and beam.id in ("ee", "mumu"))))
        if not any(process.id == self.process_id for process, _sigma in rows):
            rows.sort(key=lambda item: item[1], reverse=True)
            self.process_id = rows[0][0].id if rows else "ff13"
        rows.sort(key=lambda item: (item[0].id != self.process_id, -item[1]))
        self.rows = rows
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
            lo, hi = max(1.0, self.sqrt_s * 0.5), min(500.0, self.sqrt_s * 1.6)
        xs = np.linspace(lo, hi, 48)
        born = np.zeros_like(xs)
        radiated = np.zeros_like(xs)
        use_isr = self.isr and beam.id in ("ee", "mumu")
        for i, energy in enumerate(xs):
            born[i] = sigma_pb(process, beam, float(energy), False)
            if use_isr:
                radiated[i] = convolute_born(
                    lambda shat, p=process, b=beam: p.born_sigma(b, shat),
                    float(energy),
                    species(beam.pdg_plus).mass,
                    n=48,
                ) * PB_PER_GEV2
            else:
                radiated[i] = born[i]
        return xs, born, radiated

    def _reset_histogram(self) -> None:
        hi = max(20.0, self.sqrt_s * 1.15)
        self.mass_hist = Histogram(0.0, hi, 48)

    def spawn(self, *, animate: bool = True) -> None:
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
            self.refresh_physics()

    def run(self) -> None:
        while True:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                self._handle(event)
            self._advance_batch()
            self.anim += dt
            self.draw()
            pygame.display.flip()

    def _advance_batch(self) -> None:
        if self.batch_left <= 0:
            return
        for _ in range(4):
            if self.batch_left <= 0:
                break
            self.batch_left -= 1
            self.spawn(animate=self.batch_left == 0)

    def _handle(self, event) -> None:
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(1180, event.w), max(760, event.h)), pygame.RESIZABLE)
            return
        if event.type == pygame.MOUSEWHEEL:
            pos = pygame.mouse.get_pos()
            if self.show_methods:
                self.scroll_methods = max(0, self.scroll_methods - event.y * 28)
            elif self._zone("process").collidepoint(pos):
                self.scroll_proc = max(0, self.scroll_proc - event.y * 28)
            elif self._zone("tree").collidepoint(pos):
                self.scroll_tree = max(0, self.scroll_tree - event.y * 22)
            elif self._zone("detector").collidepoint(pos):
                factor = 0.9 if event.y > 0 else 1.1
                if self.view_3d:
                    self.orbit.distance = min(48.0, max(8.0, self.orbit.distance * factor))
                else:
                    self.view_radius = min(12.0, max(1.15, self.view_radius * factor))
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

    def _key(self, event) -> None:
        if self.energy_focus:
            if event.key == pygame.K_RETURN:
                self._commit_energy()
            elif event.key == pygame.K_ESCAPE:
                self.energy_focus = False
            elif event.key == pygame.K_BACKSPACE:
                self.energy_buffer = self.energy_buffer[:-1]
            elif event.unicode and (event.unicode.isdigit() or event.unicode == "."):
                self.energy_buffer += event.unicode
            return
        if event.key == pygame.K_ESCAPE:
            if self.show_report:
                self.show_report = False
            elif self.menu:
                self.menu = None
            elif self.picker:
                self.picker = None
            elif self.show_methods:
                self.show_methods = False
            elif self.more_open:
                self.more_open = False
            else:
                pygame.quit()
                sys.exit(0)
        elif event.key == pygame.K_SPACE:
            self.spawn()
        elif event.key == pygame.K_b:
            self.batch_total = 200
            self.batch_left = 200
        elif event.key == pygame.K_c:
            self._reset_histogram()
        elif event.key == pygame.K_F1:
            self.show_methods = not self.show_methods
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.set_energy(self.sqrt_s + (0.05 if self.range_name == "Z" else 0.5))
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.set_energy(self.sqrt_s - (0.05 if self.range_name == "Z" else 0.5))

    def _commit_energy(self) -> None:
        self.energy_focus = False
        try:
            self.set_energy(float(self.energy_buffer))
        except ValueError:
            pass

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
                if rect.collidepoint(pos) and action == "close-picker":
                    self.picker = None
                    return
            self.picker = None
            return
        if self.show_report:
            for rect, action, payload in reversed(self.hot):
                if action == "close-report" and rect.collidepoint(pos):
                    self.show_report = False
            return
        if self.show_methods:
            for rect, action, payload in reversed(self.hot):
                if action == "close-methods" and rect.collidepoint(pos):
                    self.show_methods = False
            return
        for rect, action, payload in reversed(self.hot):
            if not rect.collidepoint(pos):
                continue
            self._activate(action, payload, pos)
            return
        self.menu = None
        self.energy_focus = False
        if self.view_3d and self.zone_detector.collidepoint(pos):
            self.drag = "orbit"
            self._orbit_last = pos

    def _activate(self, action: str, payload, pos) -> None:
        if action == "menu":
            self.menu = None if self.menu == payload else str(payload)
            return
        if action == "view3d":
            self.view_3d = not self.view_3d
            self.menu = None
            return
        if action == "more":
            self.more_open = not self.more_open
            self.menu = None
            return
        if action == "section":
            key = str(payload)
            self.sections[key] = not self.sections.get(key, False)
            return
        if action == "beam":
            self.menu = None
            if payload == "custom":
                self._enter_custom()
                return
            self.custom = False
            self.picker = None
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
                self.set_energy(0.5 * (lo + hi))
        elif action == "preset":
            _label, energy, range_name = payload
            self.range_name = range_name
            self.set_energy(energy)
        elif action == "process":
            self.menu = None
            self.process_id = str(payload)
            self.refresh_physics()
        elif action == "toggle":
            setattr(self, str(payload), not getattr(self, str(payload)))
            if payload in ("isr",):
                self.refresh_physics()
        elif action == "collide":
            self.spawn()
        elif action == "batch":
            self.batch_total = 200
            self.batch_left = 200
        elif action == "clear":
            self._reset_histogram()
        elif action == "methods":
            self.show_methods = True
            self.scroll_methods = 0
        elif action == "report":
            if self.event is not None:
                self.show_report = True
                self._report_for = self.event.seed
        elif action == "energy":
            self.energy_focus = True
            self.energy_buffer = f"{self.sqrt_s:.3f}"
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
        elif self.drag == "field" and hasattr(self, "_field_rect"):
            t = (pos[0] - self._field_rect.x) / max(1, self._field_rect.w)
            self.b_field = min(4.0, max(0.0, t)) * 4.0
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

    def draw(self) -> None:
        self.hot = []
        width, height = self.screen.get_size()
        if self._bg_size != (width, height):
            self._bg = paint_background((width, height))
            self._bg_size = (width, height)
            self._static_key = None
        self.screen.blit(self._bg, (0, 0))
        self.zone_process = pygame.Rect(0, 0, 0, 0)
        self.zone_tree = pygame.Rect(0, 0, 0, 0)
        header_h = 56
        footer_h = 26
        dock_h = 64
        more_w = 372 if self.more_open else 0
        chart_h = (146 if height >= 860 else 118) if self.more_open and self.sections.get("charts") else 0
        guide_h = 168 if height >= 860 else 140
        header = pygame.Rect(0, 0, width, header_h)
        footer = pygame.Rect(0, height - footer_h, width, footer_h)
        side = pygame.Rect(width - more_w, header_h, more_w, height - header_h - footer_h)
        stage_w = side.x - 24
        dock = pygame.Rect(12, footer.y - dock_h - 8, stage_w, dock_h)
        guide = pygame.Rect(12, dock.y - guide_h - 8, stage_w, guide_h)
        cursor = guide.y - 8
        bottom = pygame.Rect(12, cursor - chart_h, stage_w, chart_h) if chart_h else pygame.Rect(0, 0, 0, 0)
        if chart_h:
            cursor = bottom.y - 8
        detector = pygame.Rect(12, header_h + 8, stage_w, max(180, cursor - header_h - 16))
        self.zone_detector = detector
        self._draw_header(header)
        self._draw_detector(detector)
        if chart_h:
            self._draw_bottom(bottom)
        else:
            self.zone_hist = pygame.Rect(0, 0, 0, 0)
            self.zone_shape = pygame.Rect(0, 0, 0, 0)
        if self.more_open:
            self._draw_side(side)
        self._draw_dock(dock)
        self._draw_guide(guide)
        self._draw_footer(footer)
        if self.menu:
            self._draw_menu()
        if self.picker:
            self._draw_picker()
        self._open_report_when_the_picture_finishes()
        if self.show_report and self.event is not None:
            self._draw_collision_report()
        if self.show_methods:
            self._draw_methods()

    def _hit(self, rect: pygame.Rect, action: str, payload=None) -> None:
        self.hot.append((rect, action, payload))

    def _draw_header(self, rect: pygame.Rect) -> None:
        pygame.draw.line(self.screen, HAIRLINE, (0, rect.bottom - 1), (rect.w, rect.bottom - 1))
        cx, cy = 36, 32
        pygame.draw.circle(self.screen, GOLD, (cx, cy), 16, 2)
        pygame.draw.circle(self.screen, (18, 46, 82), (cx, cy), 8, 2)
        pygame.draw.line(self.screen, CYAN, (cx, cy), (cx + 12, cy - 8), 2)
        title = self.fonts["title"].render("Collisions", True, TEXT)
        self.screen.blit(title, (60, 8))
        sub = self.fonts["subtitle"].render("Pick two particles, then press Collide", True, MUTED)
        self.screen.blit(sub, (62, 36))

    def _draw_detector(self, rect: pygame.Rect) -> None:
        self.zone_inset = pygame.Rect(0, 0, 0, 0)
        if self.view_3d:
            self._draw_detector_3d(rect)
            return
        camera = Camera(rect, self.view_radius)
        key = (rect.x, rect.y, rect.w, rect.h, round(self.view_radius, 3))
        if key != self._static_key:
            plate = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            local = Camera(pygame.Rect(0, 0, rect.w, rect.h), self.view_radius)
            draw_static_detector(plate, local, self.b_field)
            self._static = plate
            self._static_key = key
        halo = self._halo
        self.screen.blit(halo, halo.get_rect(center=rect.center), special_flags=pygame.BLEND_ADD)
        self.screen.blit(self._static, rect.topleft)
        previous = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-8, -8))
        approach, product = self._anim_phase()
        pair = self._approach_pair()
        show_incoming = pair is not None and (self.event is None or product <= 0.0)
        if show_incoming:
            momentum_a, momentum_b, pdg_a, pdg_b, angle = pair
            draw_incoming(
                self.screen,
                camera,
                momentum_a,
                momentum_b,
                pdg_a,
                pdg_b,
                angle,
                self.fonts["mono_small"],
                progress=0.0 if self.event is None else approach,
            )
        if self.event is not None and product > 0.0:
            # Draw into a subsurface-sized coordinate system by translating the camera,
            # which already uses screen coordinates.
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
                inset = pygame.Rect(rect.x + 16, rect.bottom - 148, 236, 132)
                self.zone_inset = inset
                draw_momentum_inset(self.screen, inset, self.event, product, self.fonts["mono_small"])
                self._hit(inset, "help-inset")
        if self.event is not None:
            self._flash(camera)
        self.screen.set_clip(previous)
        if self.more_open:
            scale = self.fonts["mono_small"].render(
                f"B = {self.b_field:.2f} T    view {self.view_radius:.2f} m",
                True,
                MUTED,
            )
            self.screen.blit(scale, (rect.x + 16, rect.y + 12))
            if self.view_radius >= 4.0:
                self._detector_legend(rect.x + 16, rect.y + 32)
        if self.more_open and self.event is not None and product > 0.0:
            name = self.fonts["ui"].render(self.event.process_title, True, GOLD)
            self.screen.blit(name, (rect.centerx - name.get_width() // 2, rect.y + 12))
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1, border_radius=16)

    def _draw_detector_3d(self, rect: pygame.Rect) -> None:
        approach, product = self._anim_phase()
        start = APPROACH_SECONDS - 0.04
        flash = 0.0
        if start < self.anim < start + 0.62:
            flash = (1.0 - (self.anim - start) / 0.62) ** 2
        previous = self.screen.get_clip()
        self.screen.set_clip(rect)
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
        self.screen.set_clip(previous)
        hint = self.fonts["mono_small"].render("Drag to turn    scroll to move closer", True, MUTED)
        self.screen.blit(hint, (rect.x + 16, rect.y + 12))
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1, border_radius=16)

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
        pygame.draw.circle(layer, (255, 244, 210, alpha), (radius + 2, radius + 2), radius)
        pygame.draw.circle(layer, (255, 255, 255, min(255, alpha + 40)), (radius + 2, radius + 2), max(4, radius // 5))
        cx, cy = camera.origin
        self.screen.blit(layer, (cx - radius - 2, cy - radius - 2), special_flags=pygame.BLEND_ADD)

    def _draw_bottom(self, rect: pygame.Rect) -> None:
        gap = 12
        left = pygame.Rect(rect.x, rect.y, rect.w // 2 - gap // 2, rect.h)
        right = pygame.Rect(left.right + gap, rect.y, rect.w - left.w - gap, rect.h)
        self.zone_hist = left
        self.zone_shape = right
        self._hit(left, "help-hist")
        self._hit(right, "help-shape")
        draw_histogram(
            self.screen,
            left,
            self.mass_hist,
            "Hard scale  √s′",
            self.fonts["ui_small"],
            self.fonts["mono_small"],
        )
        title = "Cross section"
        if self.curve is not None:
            xs, born, radiated = self.curve
            draw_lineshape(
                self.screen,
                right,
                xs,
                born,
                radiated,
                self.sqrt_s,
                title,
                self.fonts["ui_small"],
                self.fonts["mono_small"],
            )

    def _draw_side(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, PANEL, rect)
        pygame.draw.line(self.screen, HAIRLINE, (rect.x, rect.y), (rect.x, rect.bottom))
        x = rect.x + 14
        y = rect.y + 12
        w = rect.w - 28
        y, opened = self._disclosure(x, y, w, "particles", "Particles", self._pair_caption())
        if opened:
            y = self._chips(
                x,
                y,
                w,
                [(mode.label, mode.id, mode.id == self.beam_id and not self.custom) for mode in BEAMS.values()]
                + [("Any two", "custom", self.custom)],
                "beam",
            )
            if self.custom:
                y = self._draw_custom(x, y, w)
        y, opened = self._disclosure(x, y, w, "energy", "Energy", f"{self.sqrt_s:.3f} GeV")
        if opened and not self.custom:
            y = self._draw_energy(x, y, w)
        elif opened:
            note = self.fonts["ui_small"].render("Set by the two momenta below.", True, MUTED)
            self.screen.blit(note, (x, y))
            y += 22
        y, opened = self._disclosure(x, y, w, "products", "What comes out", self._outcome_caption())
        if opened:
            list_h = int(max(78, min(140, rect.bottom - y - 160)))
            list_rect = pygame.Rect(x, y, w, list_h)
            self.zone_process = list_rect
            self._process_list(list_rect)
            y += list_h + 8
        y, opened = self._disclosure(x, y, w, "detector", "Detector", f"{self.b_field:.1f} T")
        if opened:
            y = self._toggle_row(x, y, w)
            field_label = self.fonts["ui_small"].render(f"Magnetic field  {self.b_field:.2f} T", True, TEXT)
            self.screen.blit(field_label, (x, y))
            y += 16
            self._field_rect = pygame.Rect(x, y, w, 14)
            self._slider(self._field_rect, self.b_field / 4.0)
            self._hit(self._field_rect.inflate(0, 6), "bslider")
            y += 24
        y, _opened = self._disclosure(x, y, w, "charts", "Charts", "spread and rate")
        y, opened = self._disclosure(x, y, w, "event", "Every product", "energies and lifetimes")
        if opened:
            button_w = (w - 8) // 2
            batch_label = f"{self.batch_total - self.batch_left}/{self.batch_total}" if self.batch_left else "Run 200"
            self._button(pygame.Rect(x, y, button_w, 28), batch_label, "batch")
            self._button(pygame.Rect(x + button_w + 8, y, button_w, 28), "Clear", "clear")
            y += 36
            y = self._conservation(x, y, w)
            y += 6
            tree = pygame.Rect(x, y, w, max(64, rect.bottom - y - 12))
            self.zone_tree = tree
            self._tree(tree)

    def _disclosure(self, x: int, y: int, width: int, key: str, title: str, detail: str) -> tuple[int, bool]:
        opened = bool(self.sections.get(key))
        rect = pygame.Rect(x, y, width, 34)
        pygame.draw.rect(self.screen, BG_RAISED, rect, border_radius=8)
        pygame.draw.rect(self.screen, GOLD if opened else PANEL_EDGE, rect, 1, border_radius=8)
        self._chevron(x + 12, y + 13, GOLD if opened else MUTED, opened)
        label = self.fonts["ui"].render(title, True, TEXT)
        self.screen.blit(label, (x + 28, y + 7))
        detail_image = self.fonts["ui_small"].render(detail, True, MUTED)
        room = rect.right - 10 - (x + 16 + label.get_width())
        if detail_image.get_width() <= room:
            self.screen.blit(detail_image, (rect.right - detail_image.get_width() - 10, y + 9))
        self._hit(rect, "section", key)
        return y + 42, opened

    def _pair_caption(self) -> str:
        if self.custom:
            return f"{english_name(self.pdg_a)} and {english_name(self.pdg_b)}"
        beam = BEAMS[self.beam_id]
        return f"{english_name(beam.pdg_plus)} and {english_name(beam.pdg_minus)}"

    def _outcome_caption(self) -> str:
        process = process_by_id(self.process_id)
        return plain_outcome(process.id, process.title)

    def _draw_dock(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (10, 16, 30), rect, border_radius=16)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1, border_radius=16)
        gap = 8
        more = pygame.Rect(rect.right - 96, rect.y + 12, 84, 40)
        view = pygame.Rect(more.x - gap - 64, rect.y + 12, 64, 40)
        collide = pygame.Rect(view.x - gap - 120, rect.y + 12, 120, 40)
        remain = collide.x - gap - (rect.x + 10)
        pair_w = max(100, int(remain * 0.52))
        pair = pygame.Rect(rect.x + 10, rect.y + 12, pair_w, 40)
        outcome = pygame.Rect(pair.right + gap, rect.y + 12, max(80, remain - pair_w - gap), 40)
        self._menu_button(pair, self._pair_caption(), "pair")
        self._menu_button(outcome, self._outcome_caption(), "outcome")
        self._button(collide, "Collide", "collide", primary=True)
        self._button(view, "2D" if self.view_3d else "3D", "view3d", primary=self.view_3d)
        self._button(more, "Less" if self.more_open else "More", "more")

    def _chevron(self, x: int, y: int, color: tuple[int, int, int], opened: bool) -> None:
        if opened:
            points = [(x, y), (x + 10, y), (x + 5, y + 6)]
        else:
            points = [(x, y), (x + 6, y + 5), (x, y + 10)]
        pygame.draw.polygon(self.screen, color, points)

    def _menu_button(self, rect: pygame.Rect, label: str, menu_id: str) -> None:
        opened = self.menu == menu_id
        pygame.draw.rect(self.screen, (24, 40, 64) if opened else BG_RAISED, rect, border_radius=10)
        pygame.draw.rect(self.screen, GOLD if opened else PANEL_EDGE, rect, 1, border_radius=10)
        room = rect.w - 36
        shown = label
        image = self.fonts["ui"].render(shown, True, TEXT)
        while image.get_width() > room and len(shown) > 4:
            shown = shown[:-2] + "…"
            image = self.fonts["ui"].render(shown, True, TEXT)
        self.screen.blit(image, (rect.x + 12, rect.centery - image.get_height() // 2))
        self._chevron(rect.right - 18, rect.centery - 3, GOLD, opened)
        self._hit(rect, "menu", menu_id)
        if menu_id == "pair":
            self._pair_rect = rect
        else:
            self._outcome_rect = rect

    def _draw_menu(self) -> None:
        if self.menu == "pair":
            items = [
                (f"{english_name(mode.pdg_plus)} and {english_name(mode.pdg_minus)}", mode.id, mode.id == self.beam_id and not self.custom)
                for mode in BEAMS.values()
            ]
            items.append(("Choose any two…", "custom", self.custom))
            self._popover(self._pair_rect, items, "beam")
        elif self.menu == "outcome":
            items = [
                (plain_outcome(process.id, process.title), process.id, process.id == self.process_id)
                for process, _sigma in self.rows
            ]
            self._popover(self._outcome_rect, items, "process")

    def _popover(self, anchor: pygame.Rect, items: list[tuple[str, str, bool]], action: str) -> None:
        row_h = 30
        height = min(len(items) * row_h + 10, anchor.y - 16)
        rect = pygame.Rect(anchor.x, anchor.y - height - 6, max(anchor.w, 260), height)
        shadow = pygame.Surface((rect.w + 16, rect.h + 16), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 90), shadow.get_rect(), border_radius=14)
        self.screen.blit(shadow, (rect.x - 4, rect.y - 2))
        pygame.draw.rect(self.screen, (12, 20, 36), rect, border_radius=12)
        pygame.draw.rect(self.screen, GOLD, rect, 1, border_radius=12)
        y = rect.y + 5
        for label, payload, selected in items:
            row = pygame.Rect(rect.x + 5, y, rect.w - 10, row_h - 2)
            if row.bottom > rect.bottom - 4:
                break
            if selected:
                pygame.draw.rect(self.screen, (28, 48, 78), row, border_radius=7)
            image = self.fonts["ui_small"].render(label, True, GOLD if selected else TEXT)
            self.screen.blit(image, (row.x + 8, row.centery - image.get_height() // 2))
            self._hit(row, action, payload)
            y += row_h

    def _section(self, x: int, y: int, text: str) -> int:
        image = self.fonts["section"].render(text, True, GOLD)
        self.screen.blit(image, (x, y))
        return y + 20

    def _chips(self, x, y, width, items, action) -> int:
        cursor = x
        row_h = 26
        for label, payload, selected in items:
            image = self.fonts["ui_small"].render(label, True, (12, 16, 28) if selected else TEXT)
            rect = pygame.Rect(cursor, y, image.get_width() + 16, row_h)
            if rect.right > x + width:
                cursor = x
                y += row_h + 6
                rect.topleft = (cursor, y)
            color = GOLD if selected else BG_RAISED
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            pygame.draw.rect(self.screen, GOLD if selected else PANEL_EDGE, rect, 1, border_radius=6)
            self.screen.blit(image, (rect.x + 8, rect.y + 4))
            self._hit(rect, action, payload)
            cursor = rect.right + 6
        return y + row_h + 8

    def _slider(self, rect: pygame.Rect, t: float) -> None:
        pygame.draw.rect(self.screen, (20, 32, 52), rect, border_radius=6)
        fill = rect.copy()
        fill.w = int(rect.w * min(1.0, max(0.0, t)))
        pygame.draw.rect(self.screen, (36, 78, 120), fill, border_radius=6)
        knob_x = rect.x + int(rect.w * min(1.0, max(0.0, t)))
        pygame.draw.circle(self.screen, GOLD, (knob_x, rect.centery), 7)

    def _box(self, rect, edge) -> None:
        pygame.draw.rect(self.screen, (8, 12, 22), rect, border_radius=6)
        pygame.draw.rect(self.screen, edge, rect, 1, border_radius=6)

    def _process_list(self, rect: pygame.Rect) -> None:
        self._box(rect, PANEL_EDGE)
        clip = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-4, -4))
        y = rect.y + 4 - self.scroll_proc
        for process, sigma in self.rows:
            row = pygame.Rect(rect.x + 4, y, rect.w - 8, 26)
            selected = process.id == self.process_id
            if selected:
                pygame.draw.rect(self.screen, (24, 40, 64), row, border_radius=5)
            label = self.fonts["ui_small"].render(process.title, True, GOLD if selected else TEXT)
            value = self.fonts["mono_small"].render(format_cross_section(sigma), True, CYAN if selected else MUTED)
            if row.bottom > rect.y and row.y < rect.bottom:
                self.screen.blit(label, (row.x + 6, row.y + 4))
                self.screen.blit(value, (row.right - value.get_width() - 6, row.y + 5))
                self._hit(row.clip(rect), "process", process.id)
            y += 28
        self.scroll_proc = min(self.scroll_proc, max(0, y + self.scroll_proc - rect.bottom))
        self.screen.set_clip(clip)

    def _toggle_row(self, x, y, width) -> int:
        items = (
            ("Leading-log ISR", self.isr, "isr"),
            ("Decay μ", self.force_muon, "force_muon"),
            ("pT guide", self.pt_guide, "pt_guide"),
        )
        cursor = x
        for label, value, field in items:
            box = pygame.Rect(cursor, y + 2, 14, 14)
            pygame.draw.rect(self.screen, GOLD if value else (20, 32, 52), box, border_radius=3)
            pygame.draw.rect(self.screen, GOLD, box, 1, border_radius=3)
            if value:
                pygame.draw.line(self.screen, (16, 20, 28), (box.x + 3, box.y + 7), (box.x + 6, box.y + 10), 2)
                pygame.draw.line(self.screen, (16, 20, 28), (box.x + 6, box.y + 10), (box.x + 11, box.y + 3), 2)
            image = self.fonts["ui_small"].render(label, True, TEXT)
            self.screen.blit(image, (box.right + 4, y))
            hit = pygame.Rect(cursor, y, 18 + image.get_width(), 20)
            self._hit(hit, "toggle", field)
            cursor = hit.right + 8
            if cursor > x + width:
                break
        return y + 24

    def _button(self, rect, label, action, primary: bool = False) -> None:
        pygame.draw.rect(self.screen, GOLD if primary else BG_RAISED, rect, border_radius=7)
        pygame.draw.rect(self.screen, GOLD, rect, 1, border_radius=7)
        image = self.fonts["ui"].render(label, True, (16, 18, 24) if primary else TEXT)
        self.screen.blit(image, image.get_rect(center=rect.center))
        self._hit(rect, action)

    def _conservation(self, x, y, width) -> int:
        if self.report is None:
            message = self.error or "No event yet"
            self.screen.blit(self.fonts["ui_small"].render(message[:70], True, BAD if self.error else MUTED), (x, y))
            return y + 20
        self._hit(pygame.Rect(x, y, width, 32), "help-conservation")
        ok = self.report.ok
        charge = self.report.delta_charge_thirds
        charge_text = str(charge // 3) if charge % 3 == 0 else str(charge)
        header = self.fonts["ui_small"].render("Conservation  " + ("exact" if ok else "FAILED"), True, GOOD if ok else BAD)
        self.screen.blit(header, (x, y))
        detail = self.fonts["mono_small"].render(
            f"ΔE {self.report.delta_e:.1e}   Δp {self.report.delta_p:.1e}   ΔQ {charge_text}   ΔL {self.report.delta_lepton}",
            True,
            MUTED,
        )
        self.screen.blit(detail, (x, y + 16))
        return y + 32

    def _tree(self, rect: pygame.Rect) -> None:
        if rect.height < 20 or self.event is None:
            return
        self._box(rect, PANEL_EDGE)
        clip = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-6, -6))
        y = rect.y + 6 - self.scroll_tree
        seed = self.fonts["mono_small"].render(f"seed {self.event.seed}", True, DIM)
        self.screen.blit(seed, (rect.x + 8, y))
        y += 18
        for index, particle in enumerate(self.event.particles):
            if particle.status == "beam":
                continue
            depth = _depth(self.event.particles, index)
            color = MUTED if particle.status == "intermediate" else TEXT
            shown = "photon, radiated" if particle.status == "isr" else english_name(particle.pdg)
            label = self.fonts["ui_small"].render(("  " * depth) + shown, True, color)
            detail = self.fonts["mono_small"].render(brief_result(particle), True, DIM)
            if rect.y <= y <= rect.bottom - 16:
                self.screen.blit(label, (rect.x + 8, y))
                self.screen.blit(detail, (rect.right - detail.get_width() - 8, y))
            y += 18
        max_scroll = max(0, y + self.scroll_tree - rect.bottom + 8)
        self.scroll_tree = min(self.scroll_tree, max_scroll)
        self.screen.set_clip(clip)

    def _detector_legend(self, x: int, y: int) -> None:
        rows = (
            ((40, 74, 110), "tracker"),
            ((36, 78, 150), "electrons and photons"),
            ((78, 52, 36), "quarks and pions"),
            ((42, 70, 108), "muons"),
        )
        for color, text in rows:
            pygame.draw.rect(self.screen, color, pygame.Rect(x, y + 3, 10, 10), border_radius=2)
            self.screen.blit(self.fonts["mono_small"].render(text, True, MUTED), (x + 16, y))
            y += 15

    def _draw_guide(self, rect: pygame.Rect) -> None:
        self.zone_guide = rect
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1, border_radius=10)
        context = self._guide_context()
        action, payload = self._current_focus()
        focus = focus_card(action, payload, context)
        self.guide_focus_body = focus.body
        if self.more_open and action not in (None, "help-detector", "menu", "more", "section"):
            left = collision_card(context)
            right = focus
            gap = 18
            column = (rect.w - 40 - gap) // 2
            self._guide_column(rect.x + 14, rect.y + 10, column, rect.bottom - 10, left)
            divider = rect.x + 14 + column + gap // 2
            pygame.draw.line(self.screen, HAIRLINE, (divider, rect.y + 12), (divider, rect.bottom - 12))
            self._guide_column(divider + gap // 2, rect.y + 10, column, rect.bottom - 10, right)
            return
        if action not in (None, "help-detector"):
            card = focus
        elif context.approaching:
            card = type(focus)(
                "Watch",
                f"{self._pair_caption().capitalize()} start apart and move toward the center. "
                "When they meet, this panel says what was created and how long it lasts. "
                "The motion is slowed so you can follow the paths.",
            )
        elif self.event is not None and context.generated_text:
            self._draw_report(rect, self.event)
            return
        else:
            card = type(focus)(
                "Start here",
                "Choose who collides and what you want to see come out, then press Collide. "
                "More opens the energy, the magnetic field, the charts, and the full list of products.",
            )
        self._guide_column(rect.x + 16, rect.y + 10, rect.w - 32, rect.bottom - 10, card)

    def _draw_report(self, rect: pygame.Rect, event) -> None:
        cards = report_cards(event)
        gap = 8
        count = len(cards)
        width = max(120, (rect.w - 20 - gap * (count - 1)) // max(1, count))
        x = rect.x + 10
        box_h = rect.h - 16
        for title, lines in cards:
            box = pygame.Rect(x, rect.y + 8, width, box_h)
            pygame.draw.rect(self.screen, BG_RAISED, box, border_radius=8)
            pygame.draw.rect(self.screen, PANEL_EDGE, box, 1, border_radius=8)
            head = self.fonts["section"].render(title.upper(), True, GOLD)
            self.screen.blit(head, (box.x + 10, box.y + 8))
            text_y = box.y + 28
            for line in lines:
                for wrapped in self._wrap(self.fonts["ui_small"], line, box.w - 20):
                    if text_y > box.bottom - 18:
                        break
                    image = self.fonts["ui_small"].render(wrapped, True, TEXT)
                    self.screen.blit(image, (box.x + 10, text_y))
                    text_y += 16
            x += width + gap

    def _guide_column(self, x: int, y: int, width: int, bottom: int, card) -> None:
        title = self.fonts["section"].render(card.title.upper(), True, GOLD)
        self.screen.blit(title, (x, y))
        y += 18
        clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(x, y, width, max(0, bottom - y)))
        for line in self._wrap(self.fonts["ui_small"], card.body, width):
            self.screen.blit(self.fonts["ui_small"].render(line, True, TEXT), (x, y))
            y += 16
        self.screen.set_clip(clip)

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
                kind = species(particle.pdg)
                radius = curvature_radius(particle.p4.pt, kind.charge, self.b_field)
                if radius is None:
                    continue
                if best_radius is None or radius < best_radius:
                    best_radius = radius
                    tight_name = kind.name
                    tight_pt = particle.p4.pt
                    tight_radius = radius
        if self.report is not None:
            met = math.hypot(self.report.missing_px, self.report.missing_py)
        approach, product = self._anim_phase()
        approaching = self.event is not None and approach < 1.0
        generated = generated_summary(self.event) if self.event is not None and product > 0.0 else ""
        if approaching and self.event is not None:
            if self.custom:
                name_a = species(self.pdg_a).name
                name_b = species(self.pdg_b).name
                angle = self.collide_angle
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
        self.more_open = True
        self.sections["particles"] = True
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

    def _draw_energy(self, x: int, y: int, w: int) -> int:
        y = self._section(x, y, "CENTER-OF-MASS ENERGY")
        y = self._chips(
            x,
            y,
            w,
            [("0.4–30", "low", self.range_name == "low"), ("Z window", "Z", self.range_name == "Z"), ("150–500", "high", self.range_name == "high")],
            "range",
        )
        lo, hi = RANGES[self.range_name]
        self._energy_rect = pygame.Rect(x, y, w, 16)
        self._slider(self._energy_rect, (self.sqrt_s - lo) / (hi - lo))
        self._hit(self._energy_rect.inflate(0, 6), "slider")
        y += 22
        shown = self.energy_buffer if self.energy_focus else f"{self.sqrt_s:.3f} GeV"
        energy_rect = pygame.Rect(x, y, 132, 26)
        self._box(energy_rect, GOLD if self.energy_focus else PANEL_EDGE)
        self.screen.blit(self.fonts["mono"].render(shown, True, GOLD), (energy_rect.x + 8, energy_rect.y + 4))
        self._hit(energy_rect, "energy")
        return self._chips(
            energy_rect.right + 8,
            y,
            w - energy_rect.width - 8,
            [(label, (label, energy, range_name), abs(self.sqrt_s - energy) < 0.02) for label, energy, range_name in PRESETS],
            "preset",
        )

    def _draw_custom(self, x: int, y: int, w: int) -> int:
        y = self._section(x, y, "INCOMING PARTICLES")
        button_w = (w - 8) // 2
        self._particle_button(pygame.Rect(x, y, button_w, 28), self.pdg_a, "a")
        self._particle_button(pygame.Rect(x + button_w + 8, y, button_w, 28), self.pdg_b, "b")
        y += 34
        y = self._param_slider(
            x,
            y,
            w,
            f"A  {self.momentum_a:.1f} GeV   {format_beta(beta_speed(self.pdg_a, self.momentum_a))}",
            self.momentum_a / 250.0,
            "mom-a",
        )
        y = self._param_slider(
            x,
            y,
            w,
            f"B  {self.momentum_b:.1f} GeV   {format_beta(beta_speed(self.pdg_b, self.momentum_b))}",
            self.momentum_b / 250.0,
            "mom-b",
        )
        tone = "head-on" if self.collide_angle >= 175 else ("same direction" if self.collide_angle <= 5 else "between them")
        y = self._param_slider(x, y, w, f"Angle  {self.collide_angle:.0f}°   {tone}", self.collide_angle / 180.0, "angle")
        color = GOLD if self.custom_supported else WARN
        self.screen.blit(self.fonts["mono"].render(f"√s  {self.sqrt_s:.3f} GeV", True, color), (x, y))
        return y + 22

    def _particle_button(self, rect: pygame.Rect, pdg: int, slot: str) -> None:
        active = self.picker == slot
        self._box(rect, GOLD if active else PANEL_EDGE)
        image = self.fonts["ui"].render(f"{slot.upper()}   {species(pdg).name}", True, GOLD if active else TEXT)
        self.screen.blit(image, (rect.x + 8, rect.y + 4))
        self._hit(rect, "open-picker", slot)

    def _param_slider(self, x: int, y: int, w: int, text: str, t: float, action: str) -> int:
        self.screen.blit(self.fonts["mono_small"].render(text, True, TEXT), (x, y))
        rect = pygame.Rect(x, y + 16, w, 14)
        setattr(self, f"_{action.replace('-', '_')}_rect", rect)
        self._slider(rect, t)
        self._hit(rect.inflate(0, 6), action)
        return y + 36

    def _draw_picker(self) -> None:
        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((4, 6, 12, 170))
        self.screen.blit(shade, (0, 0))
        panel = pygame.Rect(0, 0, 560, 560)
        panel.center = self.screen.get_rect().center
        panel.clamp_ip(self.screen.get_rect().inflate(-24, -24))
        pygame.draw.rect(self.screen, (10, 16, 28), panel, border_radius=14)
        pygame.draw.rect(self.screen, GOLD, panel, 1, border_radius=14)
        which = "A" if self.picker == "a" else "B"
        title = self.fonts["title"].render(f"Particle {which}", True, TEXT)
        self.screen.blit(title, (panel.x + 20, panel.y + 14))
        close = pygame.Rect(panel.right - 44, panel.y + 14, 28, 28)
        pygame.draw.rect(self.screen, BG_RAISED, close, border_radius=6)
        mark = self.fonts["ui"].render("×", True, TEXT)
        self.screen.blit(mark, mark.get_rect(center=close.center))
        self._hit(close, "close-picker")
        columns = 3
        col_w = (panel.w - 48) // columns
        y = panel.y + 52
        x0 = panel.x + 20
        for group, pdgs in COLLIDER_GROUPS:
            self.screen.blit(self.fonts["section"].render(group.upper(), True, GOLD), (x0, y))
            y += 20
            for index, pdg in enumerate(pdgs):
                col = index % columns
                row = index // columns
                rect = pygame.Rect(x0 + col * col_w, y + row * 28, col_w - 8, 24)
                current = pdg == (self.pdg_a if self.picker == "a" else self.pdg_b)
                pygame.draw.rect(self.screen, (24, 40, 64) if current else (8, 12, 22), rect, border_radius=5)
                pygame.draw.rect(self.screen, GOLD if current else PANEL_EDGE, rect, 1, border_radius=5)
                self.screen.blit(self.fonts["ui_small"].render(species(pdg).name, True, TEXT), (rect.x + 8, rect.y + 3))
                self._hit(rect, "species", (self.picker, pdg))
            rows = (len(pdgs) + columns - 1) // columns
            y += rows * 28 + 8

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
        if self.zone_detector.collidepoint(pos):
            self._held_focus = ("help-detector", None)
            return self._held_focus
        self._held_focus = None
        return None, None

    def _draw_footer(self, rect: pygame.Rect) -> None:
        pygame.draw.line(self.screen, HAIRLINE, (0, rect.y), (rect.w, rect.y))
        hint = "Space collides     B runs 200     F1 formulas"
        self.screen.blit(self.fonts["mono_small"].render(hint, True, DIM), (16, rect.y + 7))
        report = pygame.Rect(rect.right - 230, rect.y + 4, 100, 22)
        methods = pygame.Rect(rect.right - 120, rect.y + 4, 100, 22)
        self._button(report, "Report", "report")
        self._button(methods, "Methods", "methods")

    def _open_report_when_the_picture_finishes(self) -> None:
        if self.event is None:
            return
        _approach, product = self._anim_phase()
        if product >= 1.0 and self._report_for != self.event.seed:
            self.show_report = True
            self._report_for = self.event.seed

    def _draw_collision_report(self) -> None:
        meet = APPROACH_SECONDS
        end = APPROACH_SECONDS + PRODUCT_PAUSE + PRODUCT_SECONDS
        timeline = energy_timeline(self.event, meet, end)
        close = draw_report_sheet(self.screen, self.fonts, self.event, timeline)
        self._hit(close, "close-report")

    def _draw_methods(self) -> None:
        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((4, 6, 12, 180))
        self.screen.blit(shade, (0, 0))
        panel = pygame.Rect(0, 0, 760, 640)
        panel.center = self.screen.get_rect().center
        pygame.draw.rect(self.screen, (10, 16, 28), panel, border_radius=14)
        pygame.draw.rect(self.screen, GOLD, panel, 1, border_radius=14)
        title = self.fonts["title"].render("Methods", True, TEXT)
        self.screen.blit(title, (panel.x + 24, panel.y + 16))
        close = pygame.Rect(panel.right - 44, panel.y + 16, 28, 28)
        pygame.draw.rect(self.screen, BG_RAISED, close, border_radius=6)
        mark = self.fonts["ui"].render("×", True, TEXT)
        self.screen.blit(mark, mark.get_rect(center=close.center))
        self._hit(close, "close-methods")
        body = panel.inflate(-48, -90)
        body.y += 28
        clip = self.screen.get_clip()
        self.screen.set_clip(body)
        y = body.y - self.scroll_methods
        for paragraph in METHODS.split("\n"):
            if not paragraph.strip():
                y += 8
                continue
            for line in self._wrap(self.fonts["ui_small"], paragraph, body.w):
                self.screen.blit(self.fonts["ui_small"].render(line, True, TEXT), (body.x, y))
                y += 18
        self.scroll_methods = min(self.scroll_methods, max(0, y + self.scroll_methods - body.bottom))
        self.screen.set_clip(clip)

    def _wrap(self, font, text: str, width: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        lines = []
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if font.size(trial)[0] <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines


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
    try:
        LabApp().run()
    except SystemExit:
        raise
    except Exception as exc:
        _report_crash(exc)
        raise


def _report_crash(exc: BaseException) -> None:
    import traceback

    text = traceback.format_exc()
    try:
        path = "smlab-error.log"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        path = ""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, f"{exc}\n\n{path}", "SMLab", 0x10)
