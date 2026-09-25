"""Perspective view of the same barrel and the same helices as the slice."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from smlab.particles import is_neutrino, species
from smlab.report import english_name
from smlab.scene import (
    ECAL,
    ECAL_HALF,
    HCAL,
    HCAL_HALF,
    MUON_HALF,
    MUONS,
    SOLENOID,
    SOLENOID_HALF,
    TRACKER_HALF,
    TRACKER_RADII,
    stop_radius,
)
from smlab.theme import GOLD, particle_color
from smlab.tracks import track_helix


@dataclass
class Orbit:
    yaw: float = 0.9
    pitch: float = 0.42
    distance: float = 16.0

    def project(self, x: float, y: float, z: float, rect: pygame.Rect) -> tuple[float, float, float] | None:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cam = (self.distance * cp * sy, self.distance * sp, self.distance * cp * cy)
        fx, fy, fz = -cam[0], -cam[1], -cam[2]
        length = math.hypot(fx, fy, fz)
        fx, fy, fz = fx / length, fy / length, fz / length
        # right = forward × world-up, with world-up along +y.
        rx, ry, rz = -fz, 0.0, fx
        rl = math.hypot(rx, ry, rz) or 1.0
        rx, ry, rz = rx / rl, ry / rl, rz / rl
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx
        vx, vy, vz = x - cam[0], y - cam[1], z - cam[2]
        depth = vx * fx + vy * fy + vz * fz
        if depth < 0.5:
            return None
        focal = 0.52 * min(rect.w, rect.h)
        sx = vx * rx + vy * ry + vz * rz
        sy = vx * ux + vy * uy + vz * uz
        return (rect.centerx + sx / depth * focal, rect.centery - sy / depth * focal, depth)


def barrel_half(pdg: int) -> float:
    """How far along the axis this species is drawn, in meters."""
    kind = abs(pdg)
    if kind in (11, 22):
        return ECAL_HALF
    if kind == 13:
        return MUON_HALF
    if is_neutrino(pdg):
        return MUON_HALF
    if kind == 21 or kind in (111, 211) or abs(species(pdg).charge) > 1.0e-6 and kind < 7:
        return HCAL_HALF
    return SOLENOID_HALF


def draw_view3d(
    surf: pygame.Surface,
    rect: pygame.Rect,
    orbit: Orbit,
    event,
    approach: float,
    product: float,
    b_field: float,
    font: pygame.font.Font,
    flash: float,
) -> None:
    """Draw the barrel and, when an event exists, the particles in it."""
    _barrel(surf, rect, orbit)
    if event is None:
        return
    if product <= 0.0:
        _incoming(surf, rect, orbit, event, approach, font)
    else:
        _products(surf, rect, orbit, event, product, b_field, font)
    if flash > 0.0:
        origin = orbit.project(0.0, 0.0, 0.0, rect)
        if origin is not None:
            radius = int(10 + (1.0 - flash) * min(rect.w, rect.h) * 0.12)
            alpha = int(160 * flash * flash)
            layer = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(layer, (255, 236, 200, alpha), (radius + 2, radius + 2), radius)
            surf.blit(layer, (origin[0] - radius - 2, origin[1] - radius - 2))


def _barrel(surf: pygame.Surface, rect: pygame.Rect, orbit: Orbit) -> None:
    layers = [(radius, TRACKER_HALF, (48, 86, 128)) for radius in TRACKER_RADII]
    layers.append((ECAL[0], ECAL_HALF, (46, 96, 178)))
    layers.append((ECAL[1], ECAL_HALF, (34, 74, 146)))
    layers.append((HCAL[0], HCAL_HALF, (122, 80, 50)))
    layers.append((HCAL[1], HCAL_HALF, (96, 62, 40)))
    layers.append((SOLENOID, SOLENOID_HALF, GOLD))
    for inner, outer in MUONS:
        layers.append((inner, MUON_HALF, (36, 58, 88)))
        layers.append((outer, MUON_HALF, (28, 48, 74)))
    lines: list[tuple[float, tuple[int, int, int], tuple[float, float], tuple[float, float]]] = []
    for radius, half, color in layers:
        _cylinder(orbit, rect, radius, -half, half, color, lines)
    # The collision axis, so the beam direction stays visible while the camera moves.
    _segment(orbit, rect, (0.0, 0.0, -SOLENOID_HALF), (0.0, 0.0, SOLENOID_HALF), (120, 98, 58), lines)
    lines.sort(key=lambda item: -item[0])
    for _depth, color, start, end in lines:
        pygame.draw.line(surf, color, start, end, 1)


def _cylinder(orbit, rect, radius, z0, z1, color, lines) -> None:
    rings = []
    for z in (z0, 0.5 * (z0 + z1), z1):
        rings.append(_ring(orbit, rect, radius, z, 28))
    for ring in rings:
        for index in range(len(ring)):
            a = ring[index]
            b = ring[(index + 1) % len(ring)]
            if a is None or b is None:
                continue
            depth = 0.5 * (a[2] + b[2])
            lines.append((depth, color, (a[0], a[1]), (b[0], b[1])))
    ends = rings[0], rings[-1]
    for spoke in range(8):
        a = ends[0][spoke * len(ends[0]) // 8]
        b = ends[1][spoke * len(ends[1]) // 8]
        if a is None or b is None:
            continue
        depth = 0.5 * (a[2] + b[2])
        lines.append((depth, color, (a[0], a[1]), (b[0], b[1])))


def _ring(orbit, rect, radius, z, count):
    points = []
    for index in range(count):
        phi = 2.0 * math.pi * index / count
        points.append(orbit.project(radius * math.cos(phi), radius * math.sin(phi), z, rect))
    return points


def _segment(orbit, rect, start, end, color, lines) -> None:
    a = orbit.project(*start, rect)
    b = orbit.project(*end, rect)
    if a is None or b is None:
        return
    lines.append((0.5 * (a[2] + b[2]), color, (a[0], a[1]), (b[0], b[1])))


def _incoming(surf, rect, orbit, event, approach: float, font) -> None:
    beams = event.beams()
    approach = max(0.0, min(1.0, approach))
    shifts = ((14, -20), (14, 10))
    for index, particle in enumerate(beams):
        momentum = particle.p4.p
        if momentum < 1.0e-6:
            point = (0.0, 0.0, 0.0)
        else:
            scale = SOLENOID_HALF * (1.0 - approach)
            point = (
                -particle.p4.px / momentum * scale,
                -particle.p4.py / momentum * scale,
                -particle.p4.pz / momentum * scale,
            )
        start = (
            -particle.p4.px / momentum * SOLENOID_HALF if momentum > 1.0e-6 else 0.0,
            -particle.p4.py / momentum * SOLENOID_HALF if momentum > 1.0e-6 else 0.0,
            -particle.p4.pz / momentum * SOLENOID_HALF if momentum > 1.0e-6 else 0.0,
        )
        color = particle_color(particle.pdg)
        _glow_line(surf, rect, orbit, start, point, color)
        projected = orbit.project(*point, rect)
        if projected is None:
            continue
        _dot(surf, projected, color)
        _name(surf, font, english_name(particle.pdg), projected, color, shifts[index % 2])


def _products(surf, rect, orbit, event, fraction: float, b_field: float, font) -> None:
    fraction = max(0.0, min(1.0, fraction))
    glow = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for particle in event.finals():
        radius = stop_radius(particle.pdg)
        if radius is None:
            continue
        kind = species(particle.pdg)
        path = track_helix(
            particle.p4.px,
            particle.p4.py,
            particle.p4.pz,
            kind.charge,
            0.0 if is_neutrino(particle.pdg) else b_field,
            radius,
            barrel_half(particle.pdg),
        )
        count = max(2, int(1 + fraction * (len(path.points) - 1)))
        shown = path.points[:count]
        screen = [orbit.project(*point, rect) for point in shown]
        color = particle_color(particle.pdg)
        _polyline(glow, screen, (*color, 80), 5)
        _polyline(surf, screen, color, 2)
        if fraction > 0.92:
            end = next((point for point in reversed(screen) if point is not None), None)
            if end is not None:
                _dot(surf, end, color)
                _name(surf, font, species(particle.pdg).name, end, color, (10, -16))
    surf.blit(glow, (0, 0))


def _glow_line(surf, rect, orbit, start, end, color) -> None:
    a = orbit.project(*start, rect)
    b = orbit.project(*end, rect)
    if a is None or b is None:
        return
    pygame.draw.line(surf, tuple(channel // 3 for channel in color), (a[0], a[1]), (b[0], b[1]), 2)


def _polyline(surf, screen, color, width: int) -> None:
    run: list[tuple[float, float]] = []
    for point in screen:
        if point is None:
            if len(run) > 1:
                pygame.draw.lines(surf, color, False, run, width)
            run = []
            continue
        run.append((point[0], point[1]))
    if len(run) > 1:
        pygame.draw.lines(surf, color, False, run, width)


def _name(surf, font, text, projected, color, shift) -> None:
    image = font.render(text, True, color)
    x = int(projected[0] + shift[0])
    y = int(projected[1] + shift[1])
    plate = pygame.Surface((image.get_width() + 8, image.get_height() + 4), pygame.SRCALPHA)
    plate.fill((6, 10, 18, 190))
    surf.blit(plate, (x - 4, y - 2))
    surf.blit(image, (x, y))


def _dot(surf, projected, color) -> None:
    center = (int(projected[0]), int(projected[1]))
    glow = pygame.Surface((28, 28), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*color, 90), (14, 14), 12)
    surf.blit(glow, (center[0] - 14, center[1] - 14))
    pygame.draw.circle(surf, color, center, 6)
    pygame.draw.circle(surf, (255, 255, 255), center, 2)


def draw_full_event_3d(surf, rect, orbit: Orbit, event, traces, fraction: float, font, flash: float) -> None:
    """The barrel and a PYTHIA event's traces (see smlab.fullscene) in perspective."""
    _barrel(surf, rect, orbit)
    fraction = max(0.0, min(1.0, fraction))
    for trace in traces:
        count = max(2, int(1 + fraction * (len(trace.points) - 1)))
        screen = [orbit.project(*point, rect) for point in trace.points[:count]]
        if trace.kind in ("photon", "decayed_neutral"):
            for i in range(0, len(screen) - 1, 2):
                a, b = screen[i], screen[i + 1]
                if a is not None and b is not None:
                    pygame.draw.line(surf, trace.color, (a[0], a[1]), (b[0], b[1]), 1)
            continue
        _polyline(surf, screen, trace.color, 2 if trace.label or trace.kind == "decayed" else 1)
        if trace.label and fraction > 0.92:
            end = next((point for point in reversed(screen) if point is not None), None)
            if end is not None:
                _name(surf, font, trace.label, end, trace.color, (8, -14))
    if flash > 0.0:
        origin = orbit.project(0.0, 0.0, 0.0, rect)
        if origin is not None:
            radius = int(10 + (1.0 - flash) * min(rect.w, rect.h) * 0.12)
            layer = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(layer, (255, 236, 200, int(160 * flash * flash)), (radius + 2, radius + 2), radius)
            surf.blit(layer, (origin[0] - radius - 2, origin[1] - radius - 2))
