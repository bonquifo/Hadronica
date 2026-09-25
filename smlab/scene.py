"""Detector, tracks, and charts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from smlab.generator import ConservationReport, Event
from smlab.histogram import Histogram
from smlab.particles import is_neutrino, is_quark, species
from smlab.report import english_name
from smlab.theme import (
    BG,
    CYAN,
    DIM,
    GOLD,
    GOLD_DIM,
    HAIRLINE,
    MUTED,
    TEXT,
    mix,
    particle_color,
)
from smlab.tracks import track_polyline

OUTER_M = 7.15
TRACKER_RADII = (0.12, 0.28, 0.48, 0.70, 0.92, 1.20)
ECAL = (1.32, 1.78)
HCAL = (1.98, 2.92)
SOLENOID = 3.05
MUONS = ((3.75, 4.25), (4.90, 5.40), (6.10, 6.75))

# 3D extent of the same schematic barrel. Inner |z| is R sinh(1.5), the
# central barrel. The solenoid half-length is the CMS coil, 12.5 m / 2.
_BARREL_SINH = math.sinh(1.5)
TRACKER_HALF = TRACKER_RADII[-1] * _BARREL_SINH
ECAL_HALF = ECAL[1] * _BARREL_SINH
HCAL_HALF = HCAL[1] * _BARREL_SINH
SOLENOID_HALF = 6.25
MUON_HALF = 6.5


@dataclass(frozen=True, slots=True)
class Camera:
    rect: pygame.Rect
    radius_m: float

    @property
    def ppm(self) -> float:
        return (min(self.rect.width, self.rect.height) * 0.46) / self.radius_m

    @property
    def origin(self) -> tuple[float, float]:
        return float(self.rect.centerx), float(self.rect.centery)

    def to_screen(self, x_m: float, y_m: float) -> tuple[float, float]:
        cx, cy = self.origin
        return cx + x_m * self.ppm, cy - y_m * self.ppm

    def radius_px(self, meters: float) -> float:
        return meters * self.ppm


_GLOW: dict[tuple, pygame.Surface] = {}


def _blit_glow(surf: pygame.Surface, color: tuple[int, int, int], center: tuple[int, int], radius: int, alpha: int) -> None:
    key = (*color, radius, alpha)
    sprite = _GLOW.get(key)
    if sprite is None:
        sprite = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            fade = int(alpha * (1.0 - r / radius) ** 2)
            pygame.draw.circle(sprite, (*color, fade), (radius, radius), r)
        _GLOW[key] = sprite
    surf.blit(sprite, (center[0] - radius, center[1] - radius))


def make_radial_glow(diameter: int, color: tuple[int, int, int], alpha: int) -> pygame.Surface:
    surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    radius = diameter // 2
    for r in range(radius, 0, -2):
        t = r / radius
        a = int(alpha * (1.0 - t) ** 2)
        pygame.draw.circle(surf, (*color, a), (radius, radius), r)
    return surf


VOID = (8, 11, 19)
# Layer colors, shared with the legend in the app.
TRACKER_COLOR = (56, 100, 150)
ECAL_COLOR = (38, 82, 160)
HCAL_COLOR = (112, 72, 46)
SOLENOID_COLOR = GOLD
MUON_COLORS = ((44, 72, 110), (34, 58, 90), (28, 48, 76))


def _disk(surf: pygame.Surface, camera: Camera, radius_m: float, color: tuple[int, int, int]) -> None:
    cx, cy = camera.origin
    radius = max(1, int(round(camera.radius_px(radius_m))))
    pygame.draw.circle(surf, color, (int(cx), int(cy)), radius)


def _sector(camera: Camera, r0: float, r1: float, phi0: float, phi1: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    steps = 5
    for i in range(steps + 1):
        phi = phi0 + (phi1 - phi0) * i / steps
        points.append(camera.to_screen(r1 * math.cos(phi), r1 * math.sin(phi)))
    for i in range(steps + 1):
        phi = phi1 - (phi1 - phi0) * i / steps
        points.append(camera.to_screen(r0 * math.cos(phi), r0 * math.sin(phi)))
    return points


def draw_static_detector(surf: pygame.Surface, camera: Camera, b_field: float) -> None:
    """Barrel geometry that does not change between events."""
    del b_field
    cx, cy = camera.origin
    outer = int(camera.radius_px(OUTER_M))
    # Filled disks, outside in. A stroked circle sits inside its radius, which
    # pulled every calorimeter band into the tracker.
    _disk(surf, camera, OUTER_M, VOID)
    pygame.draw.circle(surf, (26, 34, 52), (int(cx), int(cy)), outer, 1)

    for (r0, r1), color in zip(reversed(MUONS), MUON_COLORS):
        _disk(surf, camera, r1, color)
        _disk(surf, camera, r0, VOID)
    _disk(surf, camera, HCAL[1], HCAL_COLOR)
    _disk(surf, camera, HCAL[0], VOID)
    _disk(surf, camera, ECAL[1], ECAL_COLOR)
    _disk(surf, camera, ECAL[0], VOID)
    solenoid = max(1, int(camera.radius_px(SOLENOID)))
    pygame.draw.circle(surf, GOLD_DIM, (int(cx), int(cy)), solenoid + 2, 3)
    pygame.draw.circle(surf, SOLENOID_COLOR, (int(cx), int(cy)), solenoid, 2)

    for index in range(48):
        phi = 2.0 * math.pi * index / 48.0
        inner = camera.to_screen(ECAL[0] * math.cos(phi), ECAL[0] * math.sin(phi))
        outer_pt = camera.to_screen(ECAL[1] * math.cos(phi), ECAL[1] * math.sin(phi))
        pygame.draw.line(surf, (26, 58, 118), inner, outer_pt, 1)
    for index in range(24):
        phi = 2.0 * math.pi * index / 24.0
        inner = camera.to_screen(HCAL[0] * math.cos(phi), HCAL[0] * math.sin(phi))
        outer_pt = camera.to_screen(HCAL[1] * math.cos(phi), HCAL[1] * math.sin(phi))
        pygame.draw.line(surf, (80, 50, 32), inner, outer_pt, 1)

    for radius in TRACKER_RADII:
        pygame.draw.circle(surf, TRACKER_COLOR, (int(cx), int(cy)), max(1, int(camera.radius_px(radius))), 1)

    arm = camera.radius_px(OUTER_M - 0.08)
    pygame.draw.line(surf, (24, 32, 50), (cx - arm, cy), (cx + arm, cy), 1)
    pygame.draw.line(surf, (24, 32, 50), (cx, cy - arm), (cx, cy + arm), 1)
    pygame.draw.circle(surf, GOLD, (int(cx), int(cy)), 3)


def stop_radius(pdg: int) -> float | None:
    """Radial distance at which a final particle stops, in meters. None draws no track."""
    a = abs(pdg)
    if a in (12, 14, 16):
        return OUTER_M
    if a in (11, 22):
        return ECAL[0]
    if a == 13:
        return MUONS[-1][1]
    if a == 21 or a in (111, 211) or is_quark(pdg):
        return HCAL[0]
    return None


def deposit_layer(pdg: int) -> str | None:
    a = abs(pdg)
    if a in (11, 22):
        return "ecal"
    if a == 13:
        return "muon"
    if a == 21 or a in (111, 211) or is_quark(pdg):
        return "hcal"
    return None


def _crossing_phi(points: tuple[tuple[float, float], ...], radius_m: float) -> float:
    previous = points[0]
    for point in points[1:]:
        if math.hypot(*point) >= radius_m and math.hypot(*previous) < radius_m:
            return math.atan2(point[1], point[0])
        previous = point
    last = points[-1]
    return math.atan2(last[1], last[0])


def approach_outer(camera: Camera) -> float:
    """How far from the vertex the two particles begin, in meters."""
    return min(camera.radius_m * 0.72, 5.2)


def draw_incoming(
    surf: pygame.Surface,
    camera: Camera,
    p_a,
    p_b,
    pdg_a: int,
    pdg_b: int,
    angle_deg: float,
    font: pygame.font.Font,
    progress: float = 1.0,
) -> None:
    """Move both particles from their starts to the vertex. progress 0 is the start."""
    progress = max(0.0, min(1.0, progress))
    outer = approach_outer(camera)
    _approach_ray(surf, camera, p_a, pdg_a, outer, font, progress)
    _approach_ray(surf, camera, p_b, pdg_b, outer, font, progress)
    if p_a.p > 1.0e-6 and p_b.p > 1.0e-6 and abs(angle_deg - 180.0) > 0.5:
        label = font.render(f"{angle_deg:.0f}°", True, GOLD)
        origin = camera.to_screen(0.0, 0.0)
        surf.blit(label, (origin[0] + 12, origin[1] - 24))


def _approach_ray(surf, camera, p4, pdg: int, outer: float, font, progress: float) -> None:
    color = particle_color(pdg)
    momentum = p4.p
    if momentum < 1.0e-6:
        origin = camera.to_screen(0.0, 0.0)
        pygame.draw.circle(surf, color, (int(origin[0]), int(origin[1])), 8)
        surf.blit(font.render(f"{english_name(pdg)}, at rest", True, color), (origin[0] + 10, origin[1] - 18))
        return
    ux, uy = p4.px / momentum, p4.py / momentum
    start = camera.to_screen(-ux * outer, -uy * outer)
    end = camera.to_screen(0.0, 0.0)
    path = mix(color, (6, 10, 20), 0.62)
    pygame.draw.line(surf, path, start, end, 2)
    px = start[0] + (end[0] - start[0]) * progress
    py = start[1] + (end[1] - start[1]) * progress
    for step in range(5, 0, -1):
        back = max(0.0, progress - step * 0.012)
        tx = start[0] + (end[0] - start[0]) * back
        ty = start[1] + (end[1] - start[1]) * back
        pygame.draw.circle(surf, mix(color, (6, 10, 20), 0.45 + step * 0.08), (int(tx), int(ty)), 4)
    _blit_glow(surf, color, (int(px), int(py)), 18, 90)
    pygame.draw.circle(surf, color, (int(px), int(py)), 8)
    pygame.draw.circle(surf, (255, 255, 255), (int(px), int(py)), 3)
    text = font.render(english_name(pdg), True, color)
    # Entirely beside the dot, on the side facing the meeting point.
    if abs(ux) >= abs(uy):
        lx = px + 16 if ux >= 0 else px - 16 - text.get_width()
        ly = py - text.get_height() - 4
    else:
        lx = px + 16
        ly = py - text.get_height() / 2
    surf.blit(text, (lx, ly))


def draw_event(
    surf: pygame.Surface,
    camera: Camera,
    event: Event,
    *,
    fraction: float,
    b_field: float,
    show_guide: bool,
    font: pygame.font.Font,
    report: ConservationReport | None,
) -> None:
    fraction = max(0.0, min(1.0, fraction))
    glow = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    core = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    labels: list[tuple[tuple[float, float], str, tuple[int, int, int]]] = []

    if show_guide:
        _draw_guide(core, camera, b_field)

    for particle in event.finals():
        _draw_particle(glow, core, camera, particle.pdg, particle.p4.px, particle.p4.py, particle.p4.e, fraction, b_field, labels)

    surf.blit(glow, (0, 0))
    surf.blit(core, (0, 0))
    _place_labels(surf, labels, font, camera.rect)

    if report is not None and fraction > 0.85:
        met = math.hypot(report.missing_px, report.missing_py)
        if met > 1.0:
            _draw_met(surf, camera, report.missing_px, report.missing_py, met, font)


def _draw_guide(surf: pygame.Surface, camera: Camera, b_field: float) -> None:
    for pt, shade in ((1.0, 50), (5.0, 36), (20.0, 28)):
        path = track_polyline(pt, 0.0, 1.0, b_field, TRACKER_RADII[-1], step_m=0.03, max_turns=1.2)
        pts = [camera.to_screen(x, y) for x, y in path.points]
        if len(pts) > 1:
            pygame.draw.lines(surf, (shade, shade + 20, shade + 10), False, pts, 1)


def _draw_particle(glow, core, camera, pdg, px, py, energy, fraction, b_field, labels) -> None:
    color = particle_color(pdg)
    radius = stop_radius(pdg)
    if radius is None:
        return
    charge = species(pdg).charge
    pt = math.hypot(px, py)
    if pt < 1.0e-8:
        return
    if is_neutrino(pdg):
        path = track_polyline(px, py, 0.0, 0.0, radius)
        _dashed(core, camera, path.points, fraction, (*color, 140))
        return
    if abs(pdg) in (22, 21) or abs(charge) < 1.0e-6:
        path = track_polyline(px, py, 0.0, 0.0, radius)
    else:
        path = track_polyline(px, py, charge, b_field, radius, step_m=0.025, max_turns=3.0)
    count = max(2, int(1 + fraction * (len(path.points) - 1)))
    shown = path.points[:count]
    screen = [camera.to_screen(x, y) for x, y in shown]
    if len(screen) > 1:
        pygame.draw.lines(glow, (*color, 70), False, screen, 9)
        pygame.draw.lines(core, (*color, 245), False, screen, 2)
    if fraction > 0.92:
        end = screen[-1]
        pygame.draw.circle(glow, (*color, 90), (int(end[0]), int(end[1])), 10)
        labels.append((end, species(pdg).name, color))
    layer = deposit_layer(pdg)
    if layer and fraction > 0.55 and len(shown) > 1:
        _draw_deposit(glow, camera, layer, _crossing_phi(path.points, radius * 0.98), energy, color, fraction)


def _dashed(surf, camera, points, fraction, color) -> None:
    count = max(2, int(1 + fraction * (len(points) - 1)))
    screen = [camera.to_screen(x, y) for x, y in points[:count]]
    for index in range(0, len(screen) - 1, 2):
        pygame.draw.line(surf, color, screen[index], screen[index + 1], 1)


def _draw_deposit(surf, camera, layer, phi, energy, color, fraction) -> None:
    if layer == "ecal":
        r0, r1, bins = ECAL[0], ECAL[1], 48
    elif layer == "hcal":
        r0, r1, bins = HCAL[0], HCAL[1], 24
    else:
        r0, r1, bins = MUONS[-1][0], MUONS[-1][1], 18
    half = math.pi / bins
    # Snap to the tower that contains this azimuth.
    center = (math.floor((phi + half) / (2 * half))) * (2 * half)
    alpha = int(40 + 160 * min(1.0, math.log10(energy + 1.0) / 2.2) * fraction)
    poly = _sector(camera, r0, r1, center - half, center + half)
    pygame.draw.polygon(surf, (*color, alpha), poly)


def _draw_met(surf, camera, mpx, mpy, met, font) -> None:
    scale = camera.radius_px(min(ECAL[0], 0.15 * met))
    norm = math.hypot(mpx, mpy) or 1.0
    end = camera.to_screen(scale * mpx / norm / camera.ppm, scale * mpy / norm / camera.ppm)
    start = camera.origin
    color = (255, 176, 96)
    pygame.draw.line(surf, color, start, end, 2)
    label = font.render(f"MET {met:.1f} GeV", True, color)
    surf.blit(label, (end[0] + 6, end[1] - 8))


def _place_labels(surf, labels, font, bounds: pygame.Rect) -> None:
    placed: list[pygame.Rect] = []
    for (x, y), text, color in labels:
        image = font.render(text, True, color)
        rect = image.get_rect()
        rect.center = (int(x + 14), int(y - 12))
        for other in placed:
            if rect.colliderect(other):
                rect.y = other.bottom + 2
        rect.clamp_ip(bounds.inflate(-8, -8))
        plate = pygame.Surface((rect.width + 8, rect.height + 4), pygame.SRCALPHA)
        plate.fill((6, 10, 18, 170))
        surf.blit(plate, (rect.x - 4, rect.y - 2))
        surf.blit(image, rect)
        placed.append(rect)


def draw_momentum_inset(surf, rect: pygame.Rect, event: Event, fraction: float, font) -> None:
    """Longitudinal momentum view: beam axis (p_z) horizontal, p_y vertical."""
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, (12, 16, 27, 220), layer.get_rect(), border_radius=10)
    surf.blit(layer, rect.topleft)
    pygame.draw.rect(surf, HAIRLINE, rect, 1, border_radius=10)
    surf.blit(font.render("SIDE VIEW OF MOMENTA", True, DIM), (rect.x + 10, rect.y + 8))
    cx, cy = rect.centerx, rect.centery + 10
    pygame.draw.line(surf, (44, 52, 74), (rect.x + 14, cy), (rect.right - 14, cy), 1)
    pygame.draw.line(surf, (44, 52, 74), (cx, rect.y + 28), (cx, rect.bottom - 10), 1)
    axis = font.render("p_z →", True, DIM)
    surf.blit(axis, (rect.right - axis.get_width() - 10, cy + 3))
    surf.blit(font.render("p_y", True, DIM), (cx + 4, rect.y + 26))
    finals = event.finals()
    if not finals:
        return
    max_p = max(max(abs(p.p4.pz), p.p4.pt) for p in finals) or 1.0
    reach = min(rect.width, rect.height) * 0.36
    for particle in finals:
        color = particle_color(particle.pdg)
        z = particle.p4.pz / max_p * reach * fraction
        y = particle.p4.py / max_p * reach * fraction
        end = (cx + z, cy - y)
        pygame.draw.line(surf, color, (cx, cy), end, 2)
        pygame.draw.circle(surf, color, (int(end[0]), int(end[1])), 3)


def _chart_frame(surf, rect: pygame.Rect, title: str, subtitle: str, font, mono) -> pygame.Rect:
    pygame.draw.rect(surf, (20, 25, 38), rect, border_radius=10)
    pygame.draw.rect(surf, HAIRLINE, rect, 1, border_radius=10)
    surf.blit(font.render(title, True, TEXT), (rect.x + 12, rect.y + 8))
    surf.blit(mono.render(subtitle, True, DIM), (rect.x + 12, rect.y + 27))
    return pygame.Rect(rect.x + 62, rect.y + 52, rect.w - 76, rect.h - 76)


def _axes(surf, plot: pygame.Rect, mono, x_lo: str, x_hi: str, y_top: str) -> None:
    for i in range(1, 4):
        y = plot.bottom - plot.h * i / 4
        pygame.draw.line(surf, (32, 38, 56), (plot.x, y), (plot.right, y), 1)
    pygame.draw.line(surf, (60, 70, 98), (plot.x, plot.bottom), (plot.right, plot.bottom), 1)
    pygame.draw.line(surf, (60, 70, 98), (plot.x, plot.y), (plot.x, plot.bottom), 1)
    lo = mono.render(x_lo, True, DIM)
    hi = mono.render(x_hi, True, DIM)
    surf.blit(lo, (plot.x - lo.get_width() // 2, plot.bottom + 4))
    surf.blit(hi, (plot.right - hi.get_width(), plot.bottom + 4))
    top = mono.render(y_top, True, DIM)
    surf.blit(top, (plot.x - top.get_width() - 6, plot.y - top.get_height() // 2))
    zero = mono.render("0", True, DIM)
    surf.blit(zero, (plot.x - zero.get_width() - 6, plot.bottom - zero.get_height() // 2))


def draw_histogram(surf, rect: pygame.Rect, hist: Histogram, title: str, font, mono, unit: str = "GeV") -> None:
    mean = hist.mean()
    suffix = f" {unit}" if unit else ""
    subtitle = "one entry per event" if mean is None else f"one per event · mean {mean:.2f}{suffix}"
    plot = _chart_frame(surf, rect, title, subtitle, font, mono)
    note = mono.render(f"n = {hist.entries}", True, MUTED)
    surf.blit(note, (rect.right - note.get_width() - 96, rect.y + 11))
    if hist.total <= 0:
        _axes(surf, plot, mono, f"{hist.lo:.0f}", f"{hist.hi:.0f}{suffix}", "")
        empty = font.render("Collide or run a batch to fill it", True, DIM)
        surf.blit(empty, empty.get_rect(center=plot.center))
        return
    peak = max(hist.counts) or 1.0
    _axes(surf, plot, mono, f"{hist.lo:.0f}", f"{hist.hi:.0f}{suffix}", f"{peak:.0f}")
    bar_w = max(1, int(plot.width / hist.bins) - 1)
    for i, count in enumerate(hist.counts):
        h = int((count / peak) * plot.height)
        if h <= 0:
            continue
        x = plot.x + 1 + int(i * plot.width / hist.bins)
        bar = pygame.Rect(x, plot.bottom - h, bar_w, h)
        pygame.draw.rect(surf, (74, 144, 230), bar, border_top_left_radius=2, border_top_right_radius=2)


def draw_lineshape(surf, rect, xs, born, radiated, sqrt_s, title, font, mono) -> None:
    plot = _chart_frame(surf, rect, title, "", font, mono)
    born_color = (120, 150, 190)
    x_legend = rect.x + 12
    for label, color in (("Born", born_color), ("with ISR", GOLD), ("your √s", CYAN)):
        pygame.draw.line(surf, color, (x_legend, rect.y + 34), (x_legend + 14, rect.y + 34), 2)
        image = mono.render(label, True, MUTED)
        surf.blit(image, (x_legend + 18, rect.y + 27))
        x_legend += image.get_width() + 34
    if xs is None or len(xs) < 2:
        return
    peak = max(float(born.max()), float(radiated.max()), 1e-30)
    _axes(surf, plot, mono, f"{xs[0]:.0f}", f"{xs[-1]:.0f} GeV", _short_sigma(peak))

    def xy(i, values):
        x = plot.x + (xs[i] - xs[0]) / (xs[-1] - xs[0]) * plot.width
        y = plot.bottom - (values[i] / peak) * plot.height
        return (x, y)

    previous = surf.get_clip()
    surf.set_clip(plot.inflate(2, 2))
    pygame.draw.lines(surf, born_color, False, [xy(i, born) for i in range(len(xs))], 2)
    pygame.draw.lines(surf, GOLD, False, [xy(i, radiated) for i in range(len(xs))], 2)
    surf.set_clip(previous)
    if xs[0] <= sqrt_s <= xs[-1]:
        marker = plot.x + (sqrt_s - xs[0]) / (xs[-1] - xs[0]) * plot.width
        pygame.draw.line(surf, CYAN, (marker, plot.y), (marker, plot.bottom), 1)
        value = float(np.interp(sqrt_s, xs, radiated))
        y = plot.bottom - (value / peak) * plot.height
        pygame.draw.circle(surf, CYAN, (int(marker), int(y)), 4)


def _short_sigma(sigma_pb: float) -> str:
    if sigma_pb >= 1.0e3:
        return f"{sigma_pb / 1.0e3:.3g} nb"
    if sigma_pb >= 1.0:
        return f"{sigma_pb:.3g} pb"
    return f"{sigma_pb * 1.0e3:.3g} fb"


def paint_background(size: tuple[int, int]) -> pygame.Surface:
    """Flat application background with a faint vertical falloff."""
    width, height = size
    surf = pygame.Surface(size)
    for y in range(height):
        t = y / max(1, height - 1)
        pygame.draw.line(surf, mix(BG, (13, 17, 28), t), (0, y), (width, y))
    return surf
