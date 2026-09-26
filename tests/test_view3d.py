"""The 3D view uses the same helix as the transverse slice."""

from __future__ import annotations

import math

import pygame
import pytest

from hadronica.app import LabApp
from hadronica.tracks import curvature_radius, track_helix
from hadronica.view3d import Orbit


def test_helix_advances_along_the_field_with_the_arc():
    path = track_helix(1.0, 0.0, 2.0, 1.0, 3.8, r_stop=40.0, z_stop=40.0, step_m=0.05, max_turns=0.2)
    x, y, z = path.points[10]
    assert z == pytest.approx(2.0 * 10 * 0.05)
    assert y < 0.0
    radius = curvature_radius(1.0, 1.0, 3.8)
    assert path.radius_m == pytest.approx(radius)
    # Center sits to the right of +x for a positive charge: (0, -R).
    assert math.hypot(x, y + radius) == pytest.approx(radius, rel=1e-6)


def test_straight_track_stops_on_the_end_of_the_barrel():
    path = track_helix(0.0, 0.0, 10.0, 0.0, 3.8, r_stop=2.0, z_stop=4.0)
    x, y, z = path.points[-1]
    assert z == pytest.approx(4.0)
    assert abs(x) < 1.0e-6 and abs(y) < 1.0e-6


def test_orbit_separates_the_axis_from_the_slice():
    orbit = Orbit()
    rect = pygame.Rect(0, 0, 800, 600)
    along = orbit.project(0.0, 0.0, 6.0, rect)
    across = orbit.project(6.0, 0.0, 0.0, rect)
    assert along is not None and across is not None
    assert math.hypot(along[0] - across[0], along[1] - across[1]) > 20.0


def test_three_d_view_draws_without_changing_the_event():
    app = LabApp(size=(1480, 900), headless=True, seed=7)
    try:
        title = app.event.process_title
        seed = app.event.seed
        app.view_3d = True
        app.anim = 0.0
        app.draw()
        app.anim = 10.0
        app.draw()
        assert app.event.process_title == title
        assert app.event.seed == seed
        assert app.report.ok
    finally:
        pygame.quit()
