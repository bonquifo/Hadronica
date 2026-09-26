"""The on-screen guide has to say what a control changes, not only what it is called."""

from hadronica.guide import GuideContext, collision_card, focus_card


def _ctx(**overrides) -> GuideContext:
    base = dict(
        beam_id="ee",
        process_id="ff13",
        sqrt_s=91.188,
        shown_pb=1360.0,
        born_pb=2000.0,
        isr=True,
        force_muon=False,
        pt_guide=False,
        b_field=3.8,
        range_name="Z",
        sqrt_s_hat=91.0,
        isr_photon_energy=0.01,
        theta_deg=80.0,
        final_pdgs=(13, -13),
        tight_name="μ⁻",
        tight_pt=45.0,
        tight_radius_m=39.0,
        met=0.0,
        hist_entries=1,
    )
    base.update(overrides)
    return GuideContext(**base)


def test_z_pole_story_says_radiation_lowers_the_peak():
    card = collision_card(_ctx())
    text = card.body.lower()
    assert "1.36 nb" in card.body
    assert "2 nb" in card.body
    assert "peak" in text
    assert "collide always runs this one channel" in text
    assert "muon stations" in text


def test_above_the_z_radiation_is_called_radiative_return():
    card = collision_card(_ctx(sqrt_s=110.0, shown_pb=80.0, born_pb=40.0, range_name="Z"))
    assert "radiative return" in card.body.lower()


def test_solenoid_changes_the_picture_and_not_the_rate():
    card = focus_card("bslider", None, _ctx())
    text = card.body.lower()
    assert "clockwise" in text
    assert "does not change the cross section" in text


def test_isr_switch_names_the_effect_at_this_energy():
    card = focus_card("toggle", "isr", _ctx())
    assert "lowers" in card.body.lower() or "falls" in card.body.lower()
    assert "quark" in card.body.lower()


def test_quark_beam_explains_the_missing_proton():
    card = focus_card("beam", "uu", _ctx(beam_id="uu", isr=True))
    text = card.body.lower()
    assert "not a proton" in text
    assert "parton" in text


def test_neutrino_process_says_the_barrel_stays_empty():
    card = focus_card("process", "ff12", _ctx(process_id="ff12"))
    assert "missing-momentum" in card.body.lower()


def test_hovering_nothing_explains_how_to_read_the_screen():
    card = focus_card(None, None, _ctx())
    assert "pointer" in card.body.lower()
    assert "ecal" in card.body.lower()
