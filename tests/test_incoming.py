"""User-chosen incoming particles: momenta, angle, and the lab-frame boost."""

import math

from hadronica.constants import M_Z
from hadronica.generator import conservation_report, generate_event
from hadronica.incoming import (
    beam_for_initial,
    beta_speed,
    embed_in_lab,
    equal_headon_momentum,
    incoming_momenta,
    invariant_sqrt_s,
    rotate_z_onto,
)
from hadronica.lorentz import FourVector
from hadronica.particles import species
from hadronica.processes import process_by_id


def test_head_on_electron_momenta_recover_the_z_pole():
    momentum = equal_headon_momentum(species(11).mass, M_Z)
    pair = incoming_momenta(11, momentum, -11, momentum, 180.0)
    assert abs(invariant_sqrt_s(*pair) - M_Z) < 1.0e-6
    assert beta_speed(11, momentum) > 0.999


def test_a_smaller_angle_lowers_the_invariant_mass():
    head_on = invariant_sqrt_s(*incoming_momenta(11, 40.0, -11, 40.0, 180.0))
    right = invariant_sqrt_s(*incoming_momenta(11, 40.0, -11, 40.0, 90.0))
    glancing = invariant_sqrt_s(*incoming_momenta(11, 40.0, -11, 40.0, 0.0))
    assert right < head_on
    assert glancing < right


def test_only_matched_pairs_have_a_beam():
    beam, a_is_plus = beam_for_initial(11, -11)
    assert beam.id == "ee" and a_is_plus
    beam, a_is_plus = beam_for_initial(-11, 11)
    assert beam.id == "ee" and not a_is_plus
    assert beam_for_initial(11, 11) is None
    assert beam_for_initial(22, 11) is None
    assert beam_for_initial(13, -13)[0].id == "mumu"


def test_rotation_sends_the_beam_axis_onto_the_chosen_direction():
    moved = rotate_z_onto(FourVector(10.0, 0.0, 0.0, 4.0), (1.0, 0.0, 0.0))
    assert abs(moved.px - 4.0) < 1.0e-9
    assert abs(moved.py) < 1.0e-9
    assert abs(moved.pz) < 1.0e-9
    assert abs(moved.e - 10.0) < 1.0e-9


def test_unequal_momenta_stay_conserved_after_the_boost():
    momentum = equal_headon_momentum(species(11).mass, M_Z)
    # Keep √s on the Z, but give the positron more momentum and close the angle
    # until the invariant mass is back at the pole. A direct unequal pair is enough:
    p_a, p_b = incoming_momenta(11, momentum, -11, momentum * 1.4, 140.0)
    sqrt_s = invariant_sqrt_s(p_a, p_b)
    event = generate_event(process_by_id("ff13"), "ee", sqrt_s, seed=3, isr=True)
    embed_in_lab(event, p_a, p_b, True)
    report = conservation_report(event)
    assert report.ok
    incoming = event.beams()[0].p4 + event.beams()[1].p4
    assert abs(incoming.px - (p_a.px + p_b.px)) < 1.0e-6
    assert abs(incoming.e - (p_a.e + p_b.e)) < 1.0e-6
    # The lab is not the center of mass: total momentum is nonzero.
    assert incoming.p > 1.0


def test_photon_speed_is_c_and_a_particle_at_rest_is_not():
    assert beta_speed(22, 10.0) == 1.0
    assert beta_speed(11, 0.0) == 0.0
    assert math.isclose(beta_speed(13, species(13).mass), 1.0 / math.sqrt(2.0), rel_tol=1.0e-6)
