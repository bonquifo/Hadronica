"""Lifetimes in the collision report use PDG cτ or ħ/Γ, not the animation clock."""

from __future__ import annotations

import math

import pytest

from smlab.app import APPROACH_SECONDS, LabApp
from smlab.constants import (
    C_M_PER_S,
    C_TAU_MU_M,
    GAMMA_H,
    GAMMA_Z,
    HBAR_GEV_S,
    M_MU,
    M_Z,
    TAU_PI0_S,
)
from smlab.generator import generate_event
from smlab.guide import collision_card
from smlab.processes import BEAMS, process_by_id
from smlab.report import (
    generated_summary,
    lifetime_kind,
    mean_flight_m,
    energy_timeline,
    plain_story,
    proper_lifetime_s,
    report_cards,
)


def test_muon_proper_life_is_ctau_over_c():
    life = proper_lifetime_s(13)
    assert life == pytest.approx(C_TAU_MU_M / C_M_PER_S, rel=1e-12)
    assert proper_lifetime_s(-13) == life


def test_z_and_higgs_lifetimes_are_hbar_over_width():
    assert proper_lifetime_s(23) == pytest.approx(HBAR_GEV_S / GAMMA_Z, rel=1e-12)
    assert proper_lifetime_s(25) == pytest.approx(HBAR_GEV_S / GAMMA_H, rel=1e-12)


def test_pi0_uses_the_pdg_mean_life():
    assert proper_lifetime_s(111) == TAU_PI0_S


def test_partons_and_stable_particles_are_not_given_a_made_up_life():
    assert lifetime_kind(2) == "parton"
    assert lifetime_kind(21) == "parton"
    assert proper_lifetime_s(1) is None
    assert lifetime_kind(11) == "stable"
    assert lifetime_kind(22) == "stable"
    assert lifetime_kind(12) == "stable"


def test_muon_flight_is_beta_gamma_ctau():
    energy = M_Z / 2.0
    momentum = math.sqrt(energy * energy - M_MU * M_MU)
    flight = mean_flight_m(13, momentum, M_MU)
    assert flight == pytest.approx((momentum / M_MU) * C_TAU_MU_M, rel=1e-9)
    assert flight > 100.0e3  # hundreds of kilometers, far outside the detector


def test_energy_is_constant_until_the_particles_meet_and_then_conserved():
    event = generate_event(process_by_id("ff13"), "ee", M_Z, 3, isr=False)
    timeline = energy_timeline(event, meet_s=2.2, end_s=4.43)
    before = sum(item.energy for item in timeline.carriers_at(0.0))
    after = sum(item.energy for item in timeline.carriers_at(timeline.end_s))
    assert before == pytest.approx(timeline.total, rel=1e-9)
    assert after == pytest.approx(timeline.total, rel=1e-6)
    assert {item.name for item in timeline.before} == {"electron", "positron"}
    assert "muon" in {item.name for item in timeline.after}
    assert "No photon" in timeline.radiation
    assert "wave" in timeline.radiation
    for item in timeline.before + timeline.after:
        assert item.rest + item.kinetic == pytest.approx(item.energy, rel=1e-9)


def test_plain_story_names_the_particles_and_a_lifetime():
    event = generate_event(process_by_id("ff13"), "ee", M_Z, 3, isr=False)
    text = plain_story(event)
    assert "electron" in text and "positron" in text
    assert "muon" in text
    assert "slowed" in text
    assert "μs" in text
    titles = [title for title, _lines in report_cards(event)]
    assert "Energy" in titles and "How long" in titles
    energy = dict(report_cards(event))["Energy"][0]
    assert "GeV" in energy


def test_dimuon_report_names_the_products_and_the_hard_scatter():
    event = generate_event(process_by_id("ff13"), "ee", M_Z, 3, isr=False)
    text = generated_summary(event)
    assert "e⁻ × e⁺" in text
    assert "μ" in text
    assert "slowed" in text
    assert "left undecayed" in text
    assert "Hard scatter" in text


def test_pair_labels_say_which_particles_meet():
    assert BEAMS["ee"].label == "e⁻ × e⁺"
    assert BEAMS["mumu"].label == "μ⁻ × μ⁺"
    assert BEAMS["uu"].label == "u × ū"
    assert "beam" not in BEAMS["ee"].label.lower()


def test_collide_starts_the_approach_and_then_reports(app_display):
    app = app_display
    app.anim = 0.0
    opening = collision_card(app._guide_context())
    assert opening.title == "Approaching"
    assert "180°" in opening.body
    assert "slowed" in opening.body
    app.anim = APPROACH_SECONDS / 2.0
    halfway, products = app._anim_phase()
    assert halfway == pytest.approx(0.5)
    assert products == 0.0
    app.anim = 10.0
    done = collision_card(app._guide_context())
    assert done.title == "Generated"
    assert "μ" in done.body
    _, products = app._anim_phase()
    assert products == 1.0


@pytest.fixture
def app_display():
    import pygame

    application = LabApp(size=(1480, 900), headless=True, seed=7)
    yield application
    pygame.quit()
