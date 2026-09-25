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


def path_from_vertex(
    vertex_m: tuple[float, float, float],
    px: float,
    py: float,
    pz: float,
    charge: float,
    b_tesla: float,
    r_stop: float,
    z_stop: float,
    *,
    max_length: float | None = None,
    step_m: float = 0.05,
    max_angle_step: float = 0.08,
    max_turns: float = 2.5,
) -> list[tuple[float, float, float]]:
    """Helix (or line) from an arbitrary production vertex, in meters.

    Stops at the cylinder radius ``r_stop``, at ``|z| = z_stop``, after a path
    length ``max_length`` (a decay), or after ``max_turns`` loops. The
    transverse motion is an exact circle of radius p_T / (0.2998 |q| B); the
    motion along the field is (p_z / p_T) times the transverse arc length.
    """
    x0, y0, z0 = vertex_m
    momentum = math.sqrt(px * px + py * py + pz * pz)
    if momentum < 1.0e-12:
        return [(x0, y0, z0)]
    limit = math.inf if max_length is None else max(0.0, max_length)
    pt = math.hypot(px, py)
    radius = curvature_radius(pt, charge, b_tesla)
    points = [(x0, y0, z0)]
    if radius is None or radius > 1.0e5 or pt < 1.0e-9:
        ux, uy, uz = px / momentum, py / momentum, pz / momentum
        length = limit
        # Distance to the cylinder wall: solve |(x0,y0) + t (ux,uy)| = r_stop.
        a = ux * ux + uy * uy
        if a > 1.0e-14:
            b = x0 * ux + y0 * uy
            c = x0 * x0 + y0 * y0 - r_stop * r_stop
            disc = b * b - a * c
            if disc >= 0.0:
                t_wall = (-b + math.sqrt(disc)) / a
                if t_wall > 0.0:
                    length = min(length, t_wall)
        if abs(uz) > 1.0e-12:
            t_end = ((z_stop if uz > 0 else -z_stop) - z0) / uz
            if t_end > 0.0:
                length = min(length, t_end)
        if not math.isfinite(length):
            return points
        steps = max(1, int(length / 0.25))
        for i in range(1, steps + 1):
            t = length * i / steps
            points.append((x0 + ux * t, y0 + uy * t, z0 + uz * t))
        return points

    tx, ty = px / pt, py / pt
    if charge * b_tesla > 0.0:
        nx, ny, rot = ty, -tx, -1.0
    else:
        nx, ny, rot = -ty, tx, 1.0
    cx, cy = x0 + radius * nx, y0 + radius * ny
    rvx, rvy = x0 - cx, y0 - cy
    dphi = min(max_angle_step, step_m / radius) * rot
    cos_d, sin_d = math.cos(dphi), math.sin(dphi)
    arc_step = abs(dphi) * radius
    z_per_arc = pz / pt
    length_per_arc = momentum / pt
    arc = 0.0
    max_arc = max_turns * 2.0 * math.pi * radius
    while arc < max_arc:
        rvx, rvy = rvx * cos_d - rvy * sin_d, rvx * sin_d + rvy * cos_d
        arc += arc_step
        x, y, z = cx + rvx, cy + rvy, z0 + z_per_arc * arc
        if arc * length_per_arc >= limit:
            frac = 1.0 - (arc * length_per_arc - limit) / (arc_step * length_per_arc)
            px_, py_, pz_ = points[-1]
            points.append((px_ + (x - px_) * frac, py_ + (y - py_) * frac, pz_ + (z - pz_) * frac))
            break
        if math.hypot(x, y) >= r_stop or abs(z) >= z_stop:
            # End exactly on the layer: interpolate within the last step.
            px_, py_, pz_ = points[-1]
            frac = 1.0
            r_prev, r_now = math.hypot(px_, py_), math.hypot(x, y)
            if r_now >= r_stop and r_now > r_prev:
                frac = min(frac, (r_stop - r_prev) / (r_now - r_prev))
            if abs(z) >= z_stop and abs(z) > abs(pz_):
                frac = min(frac, (z_stop - abs(pz_)) / (abs(z) - abs(pz_)))
            frac = max(0.0, min(1.0, frac))
            points.append((px_ + (x - px_) * frac, py_ + (y - py_) * frac, pz_ + (z - pz_) * frac))
            break
        points.append((x, y, z))
    return points
