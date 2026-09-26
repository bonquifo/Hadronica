"""Mean lives and the post-collision report.

Times come from PDG cτ or from ħ/Γ. Nothing here is the length of the
animation. Quarks other than the top, and gluons, are partons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hadronica.constants import (
    C_M_PER_S,
    C_TAU_MU_M,
    C_TAU_PI_M,
    C_TAU_TAU_M,
    GAMMA_A1,
    GAMMA_H,
    GAMMA_RHO,
    GAMMA_RHO0,
    GAMMA_T,
    GAMMA_W,
    GAMMA_Z,
    HBAR_GEV_S,
    TAU_PI0_S,
    format_energy,
)
from hadronica.particles import is_quark, species

# PDG widths, GeV. The Higgs entry is the SM prediction of the LHC HXSWG.
_WIDTH_GEV = {
    6: GAMMA_T,
    23: GAMMA_Z,
    24: GAMMA_W,
    25: GAMMA_H,
    113: GAMMA_RHO0,
    213: GAMMA_RHO,
    20213: GAMMA_A1,
}

# PDG cτ, meters.
_CTAU_M = {
    13: C_TAU_MU_M,
    15: C_TAU_TAU_M,
    211: C_TAU_PI_M,
}


# Species that the built-in catalog does not know (hadrons from PYTHIA), keyed by
# PDG id: (symbol, nominal mass in GeV, proper cτ in mm) from PYTHIA's particle table.
_EXTRA_SPECIES: dict[int, tuple[str, float, float]] = {}


def register_species(mapping: dict) -> None:
    """Record names, masses, and cτ supplied by the PYTHIA worker."""
    for key, (name, mass, ctau_mm) in mapping.items():
        _EXTRA_SPECIES[int(key)] = (str(name), float(mass), float(ctau_mm))


def symbol(pdg: int) -> str:
    try:
        return species(pdg).name
    except KeyError:
        extra = _EXTRA_SPECIES.get(pdg)
        return extra[0] if extra else f"PDG {pdg}"


def rest_mass(pdg: int) -> float:
    try:
        return species(pdg).mass
    except KeyError:
        extra = _EXTRA_SPECIES.get(pdg)
        return extra[1] if extra else 0.0


def proper_lifetime_s(pdg: int) -> float | None:
    """Proper mean life in seconds, or None when the species has none here.

    PDG 2026 values for the species in constants.py; PYTHIA's particle table
    for every other hadron.
    """
    kind = abs(pdg)
    if kind == 111:
        return TAU_PI0_S
    if kind in _CTAU_M:
        return _CTAU_M[kind] / C_M_PER_S
    if kind in _WIDTH_GEV:
        return HBAR_GEV_S / _WIDTH_GEV[kind]
    extra = _EXTRA_SPECIES.get(pdg) or _EXTRA_SPECIES.get(-pdg)
    if extra and extra[2] > 0.0:
        return extra[2] * 1.0e-3 / C_M_PER_S
    return None


def _decayed(particle) -> bool:
    status = particle.status
    if isinstance(status, str):
        return status == "intermediate"
    return status < 0


def lifetime_kind(pdg: int) -> str:
    """'timed', 'parton', or 'stable'."""
    if proper_lifetime_s(pdg) is not None:
        return "timed"
    kind = abs(pdg)
    if kind == 21 or (is_quark(pdg) and kind != 6):
        return "parton"
    return "stable"


def lab_mean_life_s(pdg: int, energy: float, mass: float) -> float | None:
    """Dilated mean life γτ for this energy. None when there is no proper life."""
    proper = proper_lifetime_s(pdg)
    if proper is None or mass <= 0.0 or energy <= 0.0:
        return proper
    return (energy / mass) * proper


def mean_flight_m(pdg: int, momentum: float, mass: float) -> float | None:
    """Mean decay length βγ cτ. None when there is no proper life."""
    proper = proper_lifetime_s(pdg)
    if proper is None or mass <= 0.0:
        return None
    return (momentum / mass) * proper * C_M_PER_S


def format_duration(seconds: float) -> str:
    magnitude = abs(seconds)
    if magnitude == 0.0:
        return "0 s"
    named = ((1.0, "s"), (1e-3, "ms"), (1e-6, "μs"), (1e-9, "ns"), (1e-12, "ps"), (1e-15, "fs"))
    for scale, unit in named:
        if magnitude >= scale:
            return f"{seconds / scale:.3g} {unit}"
    exponent = math.floor(math.log10(magnitude))
    mantissa = seconds / (10.0 ** exponent)
    return f"{mantissa:.2f} × 10{_superscript(exponent)} s"


def _superscript(number: int) -> str:
    return str(number).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def format_length(meters: float) -> str:
    magnitude = abs(meters)
    if magnitude >= 1.0e3:
        return f"{meters / 1.0e3:.3g} km"
    if magnitude >= 1.0:
        return f"{meters:.3g} m"
    if magnitude >= 1.0e-3:
        return f"{meters * 1.0e3:.3g} mm"
    if magnitude >= 1.0e-6:
        return f"{meters * 1.0e6:.3g} μm"
    if magnitude >= 1.0e-9:
        return f"{meters * 1.0e9:.3g} nm"
    return f"{meters:.2e} m"


def lifetime_tag(pdg: int, energy: float, momentum: float, mass: float) -> str:
    """Short label for the event list."""
    kind = lifetime_kind(pdg)
    if kind == "parton":
        return "parton"
    if kind == "stable":
        return "stable"
    lab = lab_mean_life_s(pdg, energy, mass)
    if lab is None:
        return "stable"
    return format_duration(lab)


_ENGLISH = {
    11: "electron",
    -11: "positron",
    12: "electron neutrino",
    -12: "electron antineutrino",
    13: "muon",
    -13: "antimuon",
    14: "muon neutrino",
    -14: "muon antineutrino",
    15: "tau",
    -15: "antitau",
    16: "tau neutrino",
    -16: "tau antineutrino",
    1: "down quark",
    -1: "anti-down quark",
    2: "up quark",
    -2: "anti-up quark",
    3: "strange quark",
    -3: "anti-strange quark",
    4: "charm quark",
    -4: "anti-charm quark",
    5: "bottom quark",
    -5: "anti-bottom quark",
    6: "top quark",
    -6: "anti-top quark",
    21: "gluon",
    22: "photon",
    23: "Z boson",
    24: "W boson",
    -24: "W boson",
    25: "Higgs boson",
    111: "neutral pion",
    211: "charged pion",
    -211: "charged pion",
}


_EXTRA_ENGLISH = {
    2212: "proton",
    2112: "neutron",
    321: "charged kaon",
    130: "neutral kaon (long-lived)",
    310: "neutral kaon (short-lived)",
    3122: "lambda baryon",
    3222: "sigma baryon",
    3112: "sigma baryon",
    3312: "xi baryon",
    3334: "omega baryon",
    411: "D meson",
    421: "D meson",
    431: "D_s meson",
    511: "B meson",
    521: "B meson",
    531: "B_s meson",
    4122: "charmed lambda",
    5122: "bottom lambda",
    221: "eta meson",
    331: "eta-prime meson",
    223: "omega meson",
    333: "phi meson",
}


def english_name(pdg: int) -> str:
    """A name a non-specialist can read. The symbol stays available on the species."""
    return _ENGLISH.get(pdg) or _EXTRA_ENGLISH.get(abs(pdg)) or symbol(pdg)


def plain_outcome(process_id: str, title: str) -> str:
    special = {
        "bhabha": "the electrons scatter",
        "diphoton": "two photons",
        "zh": "a Z and a Higgs",
    }
    if process_id in special:
        return special[process_id]
    if process_id.startswith("ff") and process_id[2:].lstrip("-").isdigit():
        pdg = int(process_id[2:])
        return f"{english_name(pdg)} and {english_name(-pdg)}"
    return title


def report_cards(event) -> list[tuple[str, tuple[str, ...]]]:
    """Labeled facts for the result panel: what, the energy, the time, and each product."""
    beams = event.beams()
    if len(beams) == 2:
        who = f"{english_name(beams[0].pdg)} and {english_name(beams[1].pdg)}"
    else:
        who = "the two particles"
    made = plain_outcome(event.process_id, event.process_title)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, event.cos_theta))))
    cards: list[tuple[str, tuple[str, ...]]] = [
        ("Produced", (made, f"from {who}", f"at {angle:.0f}° from the incoming particle")),
        ("Energy", _energy_lines(event)),
        ("How long", _duration_lines(event)),
    ]
    products = _product_cards(event)
    cards.extend(products[:2])
    if len(products) > 2:
        cards.append(("Also", ("More was produced", "See the Products tab")))
    return cards


def plain_story(event) -> str:
    """The same facts as the result cards, in one paragraph."""
    return " ".join(f"{title}: {'. '.join(lines)}." for title, lines in report_cards(event))


@dataclass(frozen=True, slots=True)
class EnergyCarrier:
    name: str
    pdg: int
    energy: float
    rest: float

    @property
    def kinetic(self) -> float:
        return max(0.0, self.energy - self.rest)


@dataclass(frozen=True, slots=True)
class EnergyTimeline:
    """Energies before and after the meeting, on the slowed animation clock.

    Free particles keep a constant energy while they travel. The values change
    at the meeting, when the collision products replace the incoming pair.
    """

    meet_s: float
    end_s: float
    before: tuple[EnergyCarrier, ...]
    after: tuple[EnergyCarrier, ...]
    total: float
    hard_gev: float
    formed_s: float
    radiation: str

    def carriers_at(self, time_s: float) -> tuple[EnergyCarrier, ...]:
        return self.before if time_s < self.meet_s else self.after


def energy_timeline(event, meet_s: float, end_s: float) -> EnergyTimeline:
    """Lab energies of the incoming pair, then of the products and any photon."""
    before = tuple(_carrier(particle) for particle in event.beams())
    after_particles = [particle for particle in event.finals()]
    photon = event.isr_photon()
    if photon is not None:
        after_particles.append(photon)
    after = tuple(_carrier(particle) for particle in after_particles)
    total = sum(item.energy for item in before)
    shat = event.sqrt_s_hat if event.sqrt_s_hat else event.sqrt_s
    formed = HBAR_GEV_S / shat if shat > 0.0 else 0.0
    return EnergyTimeline(
        meet_s=meet_s,
        end_s=end_s,
        before=before,
        after=after,
        total=total,
        hard_gev=shat,
        formed_s=formed,
        radiation=_radiation_text(event),
    )


def _carrier(particle) -> EnergyCarrier:
    name = "radiated photon" if particle.status == "isr" else english_name(particle.pdg)
    rest = 0.0 if abs(particle.pdg) == 22 else rest_mass(particle.pdg)
    return EnergyCarrier(name, particle.pdg, particle.p4.e, rest)


def _radiation_text(event) -> str:
    photons = [particle for particle in event.particles if abs(particle.pdg) == 22 and particle.status != "beam"]
    if not photons:
        return (
            "No photon in this event. Radiation here means a photon. "
            "This program does not draw a classical wave."
        )
    if len(photons) > 4:
        finals = [p for p in photons if not _decayed(p)]
        total = sum(p.p4.e for p in finals)
        hardest = max((p.p4.e for p in finals), default=0.0)
        return (
            f"{len(finals)} photons reach the detector, carrying {format_energy(total)} in total; "
            f"the most energetic has {format_energy(hardest)}. Most come from π⁰ → γγ decays, "
            "the rest from radiation off charged particles."
        )
    parts = []
    for particle in photons:
        role = "Radiated before the hard collision" if particle.status == "isr" else "Produced photon"
        parts.append(f"{role}, {format_energy(particle.p4.e)}.")
    return " ".join(parts)


def brief_result(particle) -> str:
    """One line for the product list: energy, then what happens to it."""
    energy = format_energy(particle.p4.e)
    kind = lifetime_kind(particle.pdg)
    if kind == "parton":
        return f"{energy}   not a free particle"
    if kind == "stable":
        return f"{energy}   stable"
    mass = rest_mass(particle.pdg)
    proper = proper_lifetime_s(particle.pdg)
    if proper is None or mass <= 0.0:
        return f"{energy}   stable"
    flight = (particle.p4.p / mass) * proper * C_M_PER_S
    if _decayed(particle):
        if flight < 1.0e-3:
            return f"{energy}   decays at the collision"
        return f"{energy}   decayed in flight"
    if flight >= 10.0:
        return f"{energy}   flies {format_length(flight)}"
    if flight < 1.0e-3:
        return f"{energy}   decays at the collision"
    return f"{energy}   flies {format_length(flight)}"


def _energy_lines(event) -> tuple[str, ...]:
    shat = event.sqrt_s_hat if event.sqrt_s_hat else event.sqrt_s
    lines = [format_energy(shat), "energy of the hard collision"]
    photon = event.isr_photon()
    if photon is not None and photon.p4.e > 0.05:
        lines.append(f"a photon took {format_energy(photon.p4.e)}")
    elif abs(event.sqrt_s - shat) <= 0.05:
        lines.append("essentially the full beam energy")
    else:
        lines.append(f"the beams were {format_energy(event.sqrt_s)}")
    return tuple(lines)


def _duration_lines(event) -> tuple[str, ...]:
    shat = event.sqrt_s_hat if event.sqrt_s_hat else event.sqrt_s
    formed = format_duration(HBAR_GEV_S / shat) if shat > 0.0 else "an instant"
    return (formed, "how long the collision lasted", "the picture is slowed")


def _product_cards(event) -> list[tuple[str, tuple[str, ...]]]:
    cards: list[tuple[str, tuple[str, ...]]] = []
    seen: set[int] = set()
    for particle in event.particles:
        if particle.status != "intermediate" or abs(particle.pdg) in seen:
            continue
        if lifetime_kind(particle.pdg) == "stable":
            continue
        seen.add(abs(particle.pdg))
        cards.append(_particle_card(particle, None))
    finals = event.finals()
    used: set[int] = set()
    for index, particle in enumerate(finals):
        if index in used:
            continue
        partner = None
        for other_index, other in enumerate(finals):
            if other_index <= index or other_index in used:
                continue
            close = abs(other.p4.e - particle.p4.e) <= 0.15 * max(particle.p4.e, 1.0e-6)
            if abs(other.pdg) == abs(particle.pdg) and close:
                partner = other
                used.add(other_index)
                break
        used.add(index)
        cards.append(_particle_card(particle, partner))
    return cards


def _particle_card(particle, partner) -> tuple[str, tuple[str, ...]]:
    if partner is None:
        title = english_name(particle.pdg)
        energy = format_energy(particle.p4.e)
    else:
        title = f"{english_name(particle.pdg)} and {english_name(partner.pdg)}"
        energy = f"{format_energy(particle.p4.e)} each"
    clock, outcome = _life_and_fate(particle)
    return (title, (energy, clock, outcome))


def _life_and_fate(particle) -> tuple[str, str]:
    kind = lifetime_kind(particle.pdg)
    if kind == "parton":
        return ("No lifetime of its own", "Not a free particle")
    if kind == "stable":
        return ("Stable", _where(particle.pdg))
    mass = rest_mass(particle.pdg)
    proper = proper_lifetime_s(particle.pdg)
    if proper is None or mass <= 0.0:
        return ("Stable", _where(particle.pdg))
    clock = f"Lives {format_duration(proper)} on its own clock"
    if abs(particle.pdg) == 25:
        clock += ", predicted"
    flight = (particle.p4.p / mass) * proper * C_M_PER_S
    if _decayed(particle):
        if flight < 1.0e-3:
            return (clock, "Decays at the collision")
        return (clock, f"Decayed in flight; mean flight {format_length(flight)}")
    if abs(particle.pdg) > 100 and flight > 1.0:
        # Hadrons that survive the tracker are stopped by nuclear interactions long
        # before they would decay.
        return (clock, "Absorbed in the brown calorimeter")
    if flight >= 10.0:
        return (clock, f"Flies {format_length(flight)} and leaves the detector")
    if flight < 1.0e-3:
        return (clock, "Decays at the collision")
    return (clock, f"Flies {format_length(flight)} inside the detector")


def outcome_line(particle) -> str:
    """Where a product's energy goes, in a few words."""
    _clock, outcome = _life_and_fate(particle)
    if particle.status == "isr":
        return "left along the beam"
    return outcome


def _where(pdg: int) -> str:
    kind = abs(pdg)
    if kind in (12, 14, 16):
        return "Shows up only as missing momentum"
    if kind in (11, 22):
        return "Stops in the blue calorimeter"
    if kind == 13:
        return "Reaches the muon stations"
    if kind == 21 or is_quark(pdg) or kind in (111, 211) or kind > 100:
        return "Stops in the brown calorimeter"
    return "Decays at the collision"


def motion_text(name_a: str, name_b: str, angle_deg: float, sqrt_s_hat: float, *, same_frame: bool) -> str:
    formed = HBAR_GEV_S / sqrt_s_hat if sqrt_s_hat > 0.0 else 0.0
    view = (
        "They move in the plane of the screen, and the products stay in that frame."
        if same_frame
        else "They are drawn in the screen so the approach is visible. After they meet, the products are the view along their collision axis."
    )
    return (
        f"{name_a} and {name_b} start apart and travel at {angle_deg:.0f}° until they meet. "
        f"{view} The picture is slowed so those paths stay visible. "
        f"The hard scatter itself lasts ħ/√s′ = {format_duration(formed)}. "
        f"When they meet, the report lists each product and how long it lives."
    )


def generated_summary(event) -> str:
    """What the collision produced, and the mean life of each product."""
    shat = event.sqrt_s_hat if event.sqrt_s_hat else event.sqrt_s
    formed = HBAR_GEV_S / shat if shat > 0.0 else 0.0
    head = (
        f"{event.process_title}. √s′ = {format_energy(shat)}. "
        f"Hard scatter ħ/√s′ = {format_duration(formed)}. The picture is slowed."
    )
    clauses = [_clause(particle) for particle in event.particles if particle.status != "beam"]
    shown = clauses[:8]
    body = " ".join(shown)
    if len(clauses) > len(shown):
        body += " Further products are in the list on the right."
    higgs = any(abs(particle.pdg) == 25 for particle in event.particles)
    note = ""
    if higgs:
        note = " The Higgs mean life uses the Standard Model width 4.10 MeV (LHC Higgs Cross Section Working Group), not the loosely measured width."
    return f"{head} {body}{note}"


def _clause(particle) -> str:
    name = "γ ISR" if particle.status == "isr" else symbol(particle.pdg)
    energy = format_energy(particle.p4.e)
    kind = lifetime_kind(particle.pdg)
    if kind == "parton":
        return f"{name} {energy}, parton, no free lifetime."
    if kind == "stable":
        return f"{name} {energy}, stable."
    mass = rest_mass(particle.pdg)
    proper = proper_lifetime_s(particle.pdg)
    if mass <= 0.0 or proper is None:
        return f"{name} {energy}, stable."
    lab = (particle.p4.e / mass) * proper
    flight = (particle.p4.p / mass) * proper * C_M_PER_S
    fate = "decayed in this event" if _decayed(particle) else "left undecayed"
    return (
        f"{name} {energy}, proper {format_duration(proper)}, "
        f"mean life here {format_duration(lab)}, mean flight {format_length(flight)}, {fate}."
    )
