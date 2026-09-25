"""Charged-particle trajectories in a uniform solenoid.

The field points along +z (out of the transverse display). The radius follows
the PDG conversion

    p_T [GeV/c] = 0.299792458 * |q| * B[T] * R[m].

A positive charge with B along +z curves clockwise in the x-y plane (y up):
the Lorentz force q (v × B) on a +x velocity is along -y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smlab.constants import P_OVER_QRB


@dataclass(frozen=True, slots=True)
class TrackPath:
    points: tuple[tuple[float, float], ...]  # meters, y up, starting at the origin
    radius_m: float | None
    reached_stop: bool


def curvature_radius(pt_gev: float, charge: float, b_tesla: float) -> float | None:
    """Helix radius in meters. None if the track is straight."""
    if b_tesla <= 0.0 or abs(charge) < 1.0e-9 or pt_gev <= 0.0:
        return None
    return pt_gev / (P_OVER_QRB * abs(charge) * b_tesla)


def track_polyline(
    px: float,
    py: float,
    charge: float,
    b_tesla: float,
    r_stop: float,
    *,
    step_m: float = 0.02,
    max_turns: float = 2.5,
) -> TrackPath:
    """Transverse path from the origin until ``r_stop`` or until the helix loops.

    ``px``, ``py`` are momentum components in GeV. The path is exact circular
    motion (or a straight ray), not an Euler integration.
    """
    pt = math.hypot(px, py)
    if pt < 1.0e-12 or r_stop <= 0.0:
        return TrackPath(((0.0, 0.0),), None, False)
    radius = curvature_radius(pt, charge, b_tesla)
    if radius is None or radius > 1.0e6:
        return TrackPath(((0.0, 0.0), (r_stop * px / pt, r_stop * py / pt)), None, True)

    tx, ty = px / pt, py / pt
    # q B > 0 curves clockwise when y is up and B is along +z.
    # The center then sits to the right of the velocity: (ty, -tx).
    clockwise = charge * b_tesla > 0.0
    if clockwise:
        nx, ny = ty, -tx
        rot_sign = -1.0
    else:
        nx, ny = -ty, tx
        rot_sign = 1.0
    cx, cy = radius * nx, radius * ny
    rvx, rvy = -cx, -cy
    n_steps = max(12, int(max_turns * 2.0 * math.pi * radius / step_m))
    dphi = rot_sign * (max_turns * 2.0 * math.pi) / n_steps
    cos_d = math.cos(dphi)
    sin_d = math.sin(dphi)
    points = [(0.0, 0.0)]
    reached = False
    for _ in range(n_steps):
        rvx, rvy = rvx * cos_d - rvy * sin_d, rvx * sin_d + rvy * cos_d
        x = cx + rvx
        y = cy + rvy
        points.append((x, y))
        if math.hypot(x, y) >= r_stop:
            reached = True
            break
    return TrackPath(tuple(points), radius, reached)


@dataclass(frozen=True, slots=True)
class HelixPath:
    points: tuple[tuple[float, float, float], ...]
    radius_m: float | None


def track_helix(
    px: float,
    py: float,
    pz: float,
    charge: float,
    b_tesla: float,
    r_stop: float,
    z_stop: float,
    *,
    step_m: float = 0.04,
    max_turns: float = 2.5,
) -> HelixPath:
    """Path in the solenoid. The transverse circle matches ``track_polyline``.

    Distance along the field is (p_z / p_T) times the arc length in the
    transverse plane, which is exact for a uniform field along z. A straight
    track stops on the cylinder wall or on the end of the barrel, whichever
    it meets first.
    """
    momentum = math.hypot(px, py, pz)
    if momentum < 1.0e-12 or r_stop <= 0.0 or z_stop <= 0.0:
        return HelixPath(((0.0, 0.0, 0.0),), None)
    pt = math.hypot(px, py)
    radius = curvature_radius(pt, charge, b_tesla)
    if radius is None or radius > 1.0e6 or pt < 1.0e-8:
        ux, uy, uz = px / momentum, py / momentum, pz / momentum
        transverse = math.hypot(ux, uy)
        t_r = r_stop / transverse if transverse > 1.0e-8 else math.inf
        t_z = z_stop / abs(uz) if abs(uz) > 1.0e-8 else math.inf
        reach = min(t_r, t_z)
        if reach == math.inf:
            return HelixPath(((0.0, 0.0, 0.0),), None)
        steps = max(2, int(reach / step_m))
        points = tuple(
            (ux * reach * i / steps, uy * reach * i / steps, uz * reach * i / steps)
            for i in range(steps + 1)
        )
        return HelixPath(points, None)

    tx, ty = px / pt, py / pt
    clockwise = charge * b_tesla > 0.0
    if clockwise:
        nx, ny = ty, -tx
        rot_sign = -1.0
    else:
        nx, ny = -ty, tx
        rot_sign = 1.0
    cx, cy = radius * nx, radius * ny
    rvx, rvy = -cx, -cy
    dphi = rot_sign * (step_m / radius)
    cos_d = math.cos(dphi)
    sin_d = math.sin(dphi)
    z_per_arc = pz / pt
    max_arc = max_turns * 2.0 * math.pi * radius
    points = [(0.0, 0.0, 0.0)]
    arc = 0.0
    while arc < max_arc - 1.0e-9:
        rvx, rvy = rvx * cos_d - rvy * sin_d, rvx * sin_d + rvy * cos_d
        arc += step_m
        x = cx + rvx
        y = cy + rvy
        z = z_per_arc * arc
        radius_now = math.hypot(x, y)
        if radius_now >= r_stop or abs(z) >= z_stop:
            previous = points[-1]
            frac = 1.0
            radius_prev = math.hypot(previous[0], previous[1])
            if radius_now >= r_stop and radius_now > radius_prev:
                frac = min(frac, (r_stop - radius_prev) / (radius_now - radius_prev))
            if abs(z) >= z_stop and abs(z) > abs(previous[2]):
                frac = min(frac, (z_stop - abs(previous[2])) / (abs(z) - abs(previous[2])))
            frac = max(0.0, min(1.0, frac))
            points.append(
                (
                    previous[0] + (x - previous[0]) * frac,
                    previous[1] + (y - previous[1]) * frac,
                    previous[2] + (z - previous[2]) * frac,
                )
            )
            break
        points.append((x, y, z))
    return HelixPath(tuple(points), radius)
