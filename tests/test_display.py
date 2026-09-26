"""Headless checks that the event display lays out and stays conservative."""

from __future__ import annotations

import math

import pygame
import pytest

from hadronica.app import APPROACH_SECONDS, LabApp
from hadronica.scene import Camera, approach_outer


@pytest.fixture
def app():
    application = LabApp(size=(1480, 900), headless=True, seed=7)
    application.anim = 4.0
    yield application
    pygame.quit()


def _at(screen, origin, ppm, radius_m, phi):
    x = int(origin[0] + radius_m * math.cos(phi) * ppm)
    y = int(origin[1] - radius_m * math.sin(phi) * ppm)
    return screen.get_at((x, y))


def _actions(app):
    return {action for _rect, action, _payload in app.hot}


def test_detector_layers_sit_at_their_radii(app):
    app.draw()
    camera_rect = app.zone_detector.inflate(-2, -2)
    ppm = 0.46 * min(camera_rect.w, camera_rect.h) / app.view_radius
    origin = camera_rect.center
    phi = math.radians(32)
    tracker = _at(app.screen, origin, ppm, 0.55, phi)
    ecal = _at(app.screen, origin, ppm, 1.55, phi)
    hcal = _at(app.screen, origin, ppm, 2.45, phi)
    muon = _at(app.screen, origin, ppm, 4.00, phi)
    assert tracker[2] < 40  # dark tracker volume, not the ECAL band
    assert ecal[2] > ecal[0] + 20  # blue crystal
    assert hcal[0] > hcal[2]  # copper / absorber
    assert muon[2] > 40 and muon[0] < 50


def test_every_essential_control_is_on_screen(app):
    app.draw()
    actions = _actions(app)
    for needed in ("beam", "energy", "slider", "preset", "process", "toggle", "bslider", "collide", "batch",
                   "view", "tab", "report", "methods"):
        assert needed in actions, needed
    toggles = {payload for _rect, action, payload in app.hot if action == "toggle"}
    assert toggles == {"isr", "force_muon", "pt_guide"}


def test_selected_process_is_scrolled_into_view(app):
    app.draw()
    assert app.rows, "process list is empty"
    selected = [rect for rect, action, payload in app.hot if action == "process" and payload == app.process_id]
    assert selected and selected[0].h > 10
    assert app.zone_process.contains(selected[0])


def test_product_tree_is_visible_and_conservation_holds(app):
    app.tab = "products"
    app.draw()
    assert app.zone_tree.height >= 200
    assert app.report is not None and app.report.ok
    assert any(action == "help-tree" for _rect, action, _payload in app.hot)


def test_minimum_window_still_shows_everything(app):
    app.screen = pygame.display.set_mode((1180, 760), pygame.HIDDEN)
    app.draw()
    assert app.zone_process.height >= 100
    assert app.zone_detector.width >= 440
    assert "collide" in _actions(app)
    app.tab = "products"
    app.draw()
    assert app.zone_tree.height >= 120
    app.tab = "charts"
    app.draw()
    assert app.zone_hist.height >= 100 and app.zone_shape.height >= 100


def test_left_panel_scrolls_when_custom_controls_are_open(app):
    app._activate("beam", "custom", (0, 0))
    app.screen = pygame.display.set_mode((1180, 760), pygame.HIDDEN)
    app.draw()
    app._wheel(app.zone_left.center, -20)
    app.draw()
    assert app.scroll_left > 0
    assert any(action == "toggle" for _rect, action, _payload in app.hot)


def test_custom_collision_sets_sqrt_s_from_the_angle(app):
    app._activate("beam", "custom", (0, 0))
    assert app.custom
    assert app.report.ok
    head_on = app.sqrt_s
    app.collide_angle = 90.0
    app._apply_custom()
    assert app.sqrt_s < head_on
    app.spawn(animate=False)
    assert app.report.ok
    assert abs(app.report.missing_px) < 1.0e-4
    assert abs(app.report.missing_py) < 1.0e-4
    app.pdg_b = 11
    app._apply_custom()
    assert not app.custom_supported
    assert app.event is None
    app.draw()


def test_incoming_particles_start_apart_and_then_meet(app):
    app.anim = 0.0
    app.draw()
    camera = Camera(app.zone_detector.inflate(-2, -2), app.view_radius)
    outer = approach_outer(camera)
    start = camera.to_screen(-outer, 0.0)
    mid = camera.to_screen(-outer / 2.0, 0.0)
    start_px = app.screen.get_at((int(start[0]), int(start[1])))
    assert start_px[2] > 200
    app.anim = APPROACH_SECONDS / 2.0
    app.draw()
    moved = app.screen.get_at((int(mid[0]), int(mid[1])))
    left_behind = app.screen.get_at((int(start[0]), int(start[1])))
    assert moved[2] > 200
    assert left_behind[2] < 160


def test_hovering_isr_explains_the_effect_on_the_rate(app):
    app.anim = 4.0
    app.draw()
    target = next(rect for rect, action, payload in app.hot if action == "toggle" and payload == "isr")
    app.pointer_pos = target.center
    app.draw()
    text = app.guide_focus_body.lower()
    assert "peak" in text or "falls" in text or "lowers" in text
    assert app.zone_guide.height >= 120


def test_field_slider_is_clamped_to_four_tesla(app):
    app.draw()
    rect = app._field_rect
    app.drag = "field"
    app._drag((rect.right + 500, rect.centery))
    assert app.b_field == pytest.approx(4.0)
    app._drag((rect.x - 500, rect.centery))
    assert app.b_field == 0.0


def test_interactions_keep_four_momentum(app):
    app._activate("collide", None, (0, 0))
    assert app.report.ok
    app.process_id = "zh"
    app.set_energy(250.0)
    app.spawn(animate=False)
    assert app.report.ok
    assert app.event.process_title
    app.process_id = "bhabha"
    app.set_energy(91.188)
    app.spawn(animate=False)
    assert app.report.ok
    for tab in ("summary", "products", "charts"):
        app.tab = tab
        app.draw()
    app.show_methods = True
    app.draw()
    app.show_methods = False
    app.show_report = True
    app.draw()
    app.show_report = False
    app.view_3d = True
    app.draw()
    app.batch_total = 8
    app.batch_left = 8
    app._advance_batch()
    app._advance_batch()
    assert app.batch_left == 0
    assert app.mass_hist.entries == 1 + 8
    assert app.report.ok
