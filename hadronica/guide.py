"""Plain-language account of what each control does to the collision."""

from __future__ import annotations

import math
from dataclasses import dataclass

from hadronica.constants import M_H, M_Z, P_OVER_QRB, format_cross_section, format_energy
from hadronica.incoming import format_beta
from hadronica.particles import is_quark, species
from hadronica.processes import BEAMS

_FATE = {
    "ecal": "stop in the blue ECAL",
    "hcal": "stop in the brown HCAL",
    "muon": "reach the outer muon stations",
    "miss": "leave no deposit",
}


@dataclass(frozen=True, slots=True)
class GuideContext:
    beam_id: str
    process_id: str
    sqrt_s: float
    shown_pb: float
    born_pb: float
    isr: bool
    force_muon: bool
    pt_guide: bool
    b_field: float
    range_name: str
    sqrt_s_hat: float | None = None
    isr_photon_energy: float | None = None
    theta_deg: float | None = None
    final_pdgs: tuple[int, ...] = ()
    tight_name: str | None = None
    tight_pt: float | None = None
    tight_radius_m: float | None = None
    met: float = 0.0
    hist_entries: int = 0
    custom: bool = False
    custom_supported: bool = True
    pdg_a: int = 11
    pdg_b: int = -11
    momentum_a: float = 45.594
    momentum_b: float = 45.594
    angle_deg: float = 180.0
    beta_a: float = 1.0
    beta_b: float = 1.0
    approaching: bool = False
    motion_text: str = ""
    generated_text: str = ""


@dataclass(frozen=True, slots=True)
class GuideCard:
    title: str
    body: str


def _custom_intro() -> str:
    return (
        "Sets the two incoming particles yourself. Each slider is a momentum in GeV; "
        "the speed β = |p|/E is fixed by that momentum and the mass, and is printed beside the slider. "
        "The angle runs from 0° (same direction) to 180° (head-on). "
        "√s is whatever those 4-vectors imply, so the energy slider steps aside. "
        "Collide then generates the selected channel in that center of mass and boosts it into this lab frame."
    )


def _custom_story(ctx: GuideContext) -> str:
    name_a = species(ctx.pdg_a).name
    name_b = species(ctx.pdg_b).name
    meeting = "head-on" if ctx.angle_deg >= 175 else ("the same direction" if ctx.angle_deg <= 5 else f"{ctx.angle_deg:.0f}° apart")
    opening = (
        f"{name_a} at {ctx.momentum_a:.2f} GeV ({format_beta(ctx.beta_a)}) and "
        f"{name_b} at {ctx.momentum_b:.2f} GeV ({format_beta(ctx.beta_b)}), meeting {meeting}. "
        f"That fixes √s = {ctx.sqrt_s:.3f} GeV. "
        "A smaller angle at the same momenta lowers √s, because the momenta partly point the same way."
    )
    if not ctx.custom_supported:
        return (
            opening
            + " This pair has no Born formula here, so the rays are drawn and no products are invented. "
            "Use a particle and its antiparticle — e⁻e⁺, μ⁻μ⁺, or q q̄ — to generate a final state."
        )
    rate = _rate_effect(ctx)
    return opening + " The products are that channel, boosted into this lab. " + rate


def _momentum_body(ctx: GuideContext, first: bool) -> str:
    pdg = ctx.pdg_a if first else ctx.pdg_b
    momentum = ctx.momentum_a if first else ctx.momentum_b
    beta = ctx.beta_a if first else ctx.beta_b
    name = species(pdg).name
    return (
        f"{name} is at {momentum:.2f} GeV, so its speed is {format_beta(beta)}. "
        "Drag the slider to change the momentum. Speed moves with it: a heavier particle at the same "
        "momentum is slower, and a massless one is always at c. "
        f"Together with the other particle and the {ctx.angle_deg:.0f}° angle, this is what sets √s "
        f"({ctx.sqrt_s:.3f} GeV). Releasing the slider generates a new event."
    )


def _angle_body(ctx: GuideContext) -> str:
    return (
        f"The angle between the two momenta is {ctx.angle_deg:.0f}°. "
        "180° is head-on: the rays form a straight line through the origin and √s is as large as these momenta allow. "
        "90° is a right angle in the plane of the screen. "
        "0° means both travel the same way, the center-of-mass energy collapses toward the mass threshold, "
        "and most channels close. The products, when a formula exists, are boosted from that center of mass into this lab."
    )


def collision_card(ctx: GuideContext) -> GuideCard:
    if ctx.approaching and ctx.motion_text:
        return GuideCard("Approaching", ctx.motion_text)
    if ctx.generated_text:
        return GuideCard("Generated", ctx.generated_text)
    if ctx.custom:
        return GuideCard("This collision", _custom_story(ctx))
    beam = BEAMS[ctx.beam_id].label
    title = process_title(ctx.process_id)
    rate = format_cross_section(ctx.shown_pb)
    parts = [
        (
            f"{beam} at {ctx.sqrt_s:.3f} GeV, {title}. "
            f"The listed rate is {rate}. Collide always runs this one channel; "
            f"it does not roll a die weighted by the other cross sections."
        ),
        _rate_effect(ctx),
        _event_effect(ctx) if ctx.sqrt_s_hat is not None else _picture_effect(ctx),
    ]
    return GuideCard("This collision", " ".join(part for part in parts if part))


def focus_card(action: str | None, payload, ctx: GuideContext) -> GuideCard:
    if action is None:
        return GuideCard(
            "How to read the screen",
            "Move the pointer over a control, the detector, or either plot. "
            "This card says what that piece is and what it changes in the collision. "
            "Custom, next to the particle pairs, chooses both incoming particles, each momentum, and the angle between them. "
            "F1 opens the formulas, the sources, and the list of approximations. "
            "The rings, from the inside, are the tracker, the blue ECAL, the brown HCAL, and the muon stations.",
        )
    if action == "beam" and payload == "custom":
        return GuideCard("Custom collision", _custom_intro())
    if action == "open-picker":
        which = "A" if payload == "a" else "B"
        return GuideCard(
            f"Particle {which}",
            "Opens the species list. The choice is the incoming particle, not the thing it decays into. "
            "Products are computed only for e⁻e⁺, μ⁻μ⁺, and a quark with its own antiquark. "
            "Any other pair is drawn coming in, and no final state is invented for it.",
        )
    if action in ("mom-a", "mom-b"):
        return GuideCard("Momentum and speed", _momentum_body(ctx, action == "mom-a"))
    if action == "angle":
        return GuideCard("Collision angle", _angle_body(ctx))
    if action == "beam":
        return GuideCard("Particles", _beam_body(str(payload), ctx))
    if action == "range":
        return GuideCard("Energy window", _range_body(str(payload)))
    if action == "slider":
        return GuideCard("Energy slider", _slider_body(ctx))
    if action == "energy":
        return GuideCard("Center-of-mass energy", _energy_body(ctx))
    if action == "preset":
        _label, energy, _range = payload
        return GuideCard(f"Preset  {energy:g} GeV", _preset_body(float(energy)))
    if action == "process":
        return GuideCard(process_title(str(payload)), _process_body(str(payload), ctx))
    if action == "toggle":
        return _toggle_card(str(payload), ctx)
    if action == "bslider":
        return GuideCard("Solenoid", _solenoid_body(ctx))
    if action == "view":
        if payload == "3d":
            return focus_card("view3d", None, ctx)
        return GuideCard(
            "Transverse view",
            "Looks straight down the beam pipe, the standard event-display view. "
            "Charged tracks are circles of radius R = p_T / (0.2998 |q| B); momentum along the beam is invisible here, "
            "so the side-view inset shows it. Scroll over the detector to zoom.",
        )
    if action == "tab":
        texts = {
            "summary": "Key numbers for the event on screen: the hard-collision energy, the cross section, the primary angle, "
            "missing transverse momentum, and what was produced with its lifetime.",
            "products": "The complete decay chain of the event on screen, with lab-frame energies and what happens to each particle.",
            "charts": "The hard-scale histogram of all events so far, and the cross section of the selected channel across energy.",
        }
        return GuideCard("Results", texts.get(str(payload), "Results for the event on screen."))
    if action == "report":
        return GuideCard(
            "Full report",
            "Opens the collision report: every product with its energy and fate, and how the energy is shared "
            "before and after the particles meet. Total energy is the same on both sides.",
        )
    if action == "view3d":
        return GuideCard(
            "3D view",
            "Draws this same collision inside the barrel, which you can turn. "
            "Drag the picture to look around and scroll to move closer. "
            "A charged track is a helix in the solenoid: the circle in the slice is p_T = 0.2998 |q| B R, "
            "and the distance along the field is p_z / p_T times that arc. "
            "The button returns to the flat slice. The camera does not change what was produced.",
        )
    if action == "collide":
        return GuideCard(
            "Collide",
            "Places the two particles at their starting positions, moves them along the selected angle "
            "until they meet, then reports what was produced and each mean life. "
            "The motion is slowed. The times in that report are ħ/√s′ and PDG lifetimes, not the length of the picture. "
            "The outgoing angle follows that process's real distribution, so forward Bhabha electrons are common "
            "and Z muon pairs are not. The seed in the event list reproduces the same collision.",
        )
    if action == "batch":
        return GuideCard(
            "Run 200",
            "Generates 200 events with these same settings and adds each hard scale √s′ to the histogram. "
            "Only the last one is drawn. This is how you see the spread from radiation, "
            "rather than judging the physics from a single picture. "
            f"The histogram currently holds {ctx.hist_entries} entries.",
        )
    if action == "clear":
        return GuideCard(
            "Clear",
            "Empties the hard-scale histogram. The event on screen stays, and the cross section does not change.",
        )
    if action == "methods":
        return GuideCard(
            "Methods",
            "Opens the formulas, the PDG 2026 and CODATA 2022 inputs with their sources, and what this Born laboratory leaves out: "
            "no parton shower, no hadronization, no proton structure, and no full one-loop electroweak library.",
        )
    if action == "help-detector":
        return GuideCard("Detector", _detector_body(ctx))
    if action == "help-hist":
        return GuideCard("Hard scale", _hist_body(ctx))
    if action == "help-shape":
        return GuideCard("Cross-section curve", _shape_body(ctx))
    if action == "help-inset":
        return GuideCard(
            "Momentum inset",
            "The big view looks along the beam, so momentum down the pipe is invisible there. "
            "Here the beam axis runs left to right and the vertical momentum runs up. "
            "A balanced pair with no radiation is a straight line through the origin. "
            "A radiated photon, drawn along the beam, shifts the pair off that line.",
        )
    if action == "help-tree":
        return GuideCard(
            "Event list",
            "The decay chain of the event on screen. Dimmed rows are bosons that decayed before the detector; "
            "the bright rows are what is actually drawn. Energies are in the lab frame. "
            "Scroll this list when a Higgs or tau cascade is longer than the box. "
            "The seed regenerates this exact event.",
        )
    if action == "help-conservation":
        return GuideCard("Conservation", _conservation_body())
    return focus_card(None, None, ctx)


def process_title(process_id: str) -> str:
    names = {
        "bhabha": "Bhabha scattering, e⁻ e⁺ → e⁻ e⁺",
        "diphoton": "two-photon annihilation, e⁻ e⁺ → γ γ",
        "zh": "Higgsstrahlung, e⁻ e⁺ → Z H",
    }
    if process_id in names:
        return names[process_id]
    if process_id.startswith("ff"):
        pdg = int(process_id[2:])
        return f"{species(pdg).name} {species(-pdg).name} pair"
    return process_id


def _rate_effect(ctx: GuideContext) -> str:
    applies = ctx.isr and ctx.beam_id in ("ee", "mumu")
    if ctx.beam_id not in ("ee", "mumu"):
        return (
            "These are quark beams, so the lepton radiator is off and the rate is a parton-level "
            "Born cross section. It is not a proton-collider prediction: there is no parton distribution "
            "and no underlying event."
        )
    if not ctx.isr:
        return (
            "Initial-state radiation is off, so the hard collision uses the full beam energy "
            f"and the listed rate is the Born value, {format_cross_section(ctx.born_pb)}."
        )
    if not applies:
        return ""
    born = format_cross_section(ctx.born_pb)
    shown = format_cross_section(ctx.shown_pb)
    if ctx.born_pb <= 0:
        return "Radiation is on."
    ratio = ctx.shown_pb / ctx.born_pb
    if abs(ctx.sqrt_s - M_Z) < 3.0 and ratio < 0.95:
        return (
            f"Radiation is on, so the rate falls from {born} to {shown}. "
            "The photon usually knocks the collision off the narrow Z peak."
        )
    if ctx.sqrt_s > M_Z + 6.0 and ratio > 1.08:
        return (
            f"Radiation is on, so the rate rises from {born} to {shown}. "
            "Some photons carry away just enough energy for the hard collision to land back on the Z. "
            "That is radiative return."
        )
    return f"Radiation is on. The Born rate is {born} and the radiated rate is {shown}."


def _picture_effect(ctx: GuideContext) -> str:
    if ctx.b_field <= 0.0:
        bend = "The solenoid is off, so every charged track is straight. The field never changes the rate, only the picture."
    else:
        r1 = 1.0 / (P_OVER_QRB * ctx.b_field)
        bend = (
            f"At {ctx.b_field:.2f} T a 1 GeV track bends with radius {r1:.2f} m. "
            "The ECAL starts at 1.32 m, so a much softer track curls up in the tracker and never deposits. "
            "A 45 GeV muon from the Z has a radius of tens of meters and looks straight."
        )
    if ctx.force_muon:
        bend += " Muon decay is forced, so a muon becomes an electron plus neutrinos instead of a track to the outer stations."
    return bend


def _event_effect(ctx: GuideContext) -> str:
    sentences = []
    if ctx.theta_deg is not None and ctx.process_id not in ("diphoton",):
        sentences.append(
            f"In this event the primary product leaves at {ctx.theta_deg:.0f}° from the beam. "
            "Near 90° it crosses the barrel; near 0° it would escape down the pipe."
        )
    if ctx.isr_photon_energy is not None and ctx.isr_photon_energy > 0.05 and ctx.sqrt_s_hat is not None:
        sentences.append(
            f"A photon took {format_energy(ctx.isr_photon_energy)} down the beam pipe, "
            f"so the hard collision ran at {format_energy(ctx.sqrt_s_hat)} rather than {format_energy(ctx.sqrt_s)}."
        )
    elif ctx.isr and ctx.beam_id in ("ee", "mumu") and ctx.sqrt_s_hat is not None:
        sentences.append("This event radiated very little, so √s′ is essentially the full beam energy.")
    arrivals = _arrivals(ctx.final_pdgs)
    if arrivals:
        sentences.append(arrivals)
    if ctx.tight_name and ctx.tight_pt is not None and ctx.tight_radius_m is not None and ctx.b_field > 0:
        if ctx.tight_radius_m < 0.7:
            sentences.append(
                f"The tightest track is {ctx.tight_name} at pT {ctx.tight_pt:.2f} GeV, "
                f"radius {ctx.tight_radius_m:.2f} m, so it curls inside the tracker."
            )
        elif ctx.tight_radius_m < 8:
            sentences.append(
                f"The tightest track is {ctx.tight_name} at pT {ctx.tight_pt:.2f} GeV, "
                f"radius {ctx.tight_radius_m:.2f} m."
            )
    if ctx.met > 1.0:
        sentences.append(f"Missing transverse momentum is {ctx.met:.1f} GeV, from neutrinos.")
    return " ".join(sentences)


def _arrivals(pdgs: tuple[int, ...]) -> str:
    groups: dict[str, list[str]] = {}
    for pdg in pdgs:
        fate = _fate(pdg)
        if fate is None:
            continue
        groups.setdefault(fate, []).append(species(pdg).name)
    if not groups:
        return ""
    bits = []
    for fate, names in groups.items():
        shown = _join_names(names)
        bits.append(f"{shown} {_FATE[fate]}")
    return "In the detector, " + "; ".join(bits) + "."


def _join_names(names: list[str]) -> str:
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    if len(unique) == 1 and names.count(unique[0]) == 2:
        return f"both {unique[0]}"
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def _fate(pdg: int) -> str | None:
    a = abs(pdg)
    if a in (12, 14, 16):
        return "miss"
    if a in (11, 22):
        return "ecal"
    if a == 13:
        return "muon"
    if a == 21 or a in (111, 211) or is_quark(pdg):
        return "hcal"
    return None


def _beam_body(beam_id: str, ctx: GuideContext) -> str:
    selected = " These two are the pair in use." if beam_id == ctx.beam_id else " Choosing it rebuilds the process list and the curve."
    texts = {
        "ee": (
            "An electron and a positron. The electron travels along +z, and every angle in the formulas "
            "is measured from that direction. Lepton beams can radiate a collinear photon. "
            "Bhabha scattering and two photons are available only here."
        ),
        "mumu": (
            "A muon and an antimuon. The electroweak formulas match the electron collider, with the muon mass "
            "in the radiator, so initial-state radiation is weaker. Same-flavor muon scattering is not offered; "
            "use a muon pair as the final state on an electron beam instead."
        ),
        "uu": _quark_beam("up", "charge +2/3"),
        "dd": _quark_beam("down", "charge −1/3"),
        "ss": _quark_beam("strange", "charge −1/3"),
        "cc": _quark_beam("charm", "charge +2/3"),
        "bb": _quark_beam("bottom", "charge −1/3"),
    }
    return texts.get(beam_id, "Beam species.") + selected


def _quark_beam(flavor: str, charge: str) -> str:
    return (
        f"A {flavor} quark and its antiquark, {charge}, not a proton. "
        "The rate is parton-level annihilation into leptons or a heavier quark pair. "
        "Quark-quark scattering is omitted, the lepton radiator is off, and the channel stays closed below 8 GeV "
        "so the Born curve is not drawn on top of hadron resonances."
    )


def _range_body(name: str) -> str:
    texts = {
        "low": (
            "Lets the slider run from 0.4 to 30 GeV. Photon exchange dominates, the Z is a small correction, "
            "and charged tracks are soft enough to curl in the 3.8 T field. "
            "Higgsstrahlung and top pairs are closed."
        ),
        "Z": (
            "Lets the slider run from 60 to 140 GeV, across the Z resonance at 91.188 GeV. "
            "On the peak the hadronic channels are the largest rates, one neutrino flavor is larger than one "
            "charged lepton, and a few GeV either side drops the rate by an order of magnitude."
        ),
        "high": (
            "Lets the slider run from 150 to 500 GeV. ZH opens above about 216 GeV and peaks near 250 GeV. "
            "Top pairs open near 345 GeV. Tracks from these collisions look straight."
        ),
    }
    return texts.get(name, "Chooses the range the energy slider can reach. It does not by itself generate an event.")


def _slider_body(ctx: GuideContext) -> str:
    windows = {"low": "0.4–30 GeV", "Z": "60–140 GeV", "high": "150–500 GeV"}
    window = windows.get(ctx.range_name, "the selected window")
    return (
        f"Sets √s inside {window}, currently {ctx.sqrt_s:.3f} GeV. "
        "Moving it updates the cross sections and closes any channel that falls below threshold. "
        "It does not throw away the histogram until the new energy no longer fits the plotted window. "
        "The next Collide uses the new energy: higher √s usually means straighter tracks."
    )


def _energy_body(ctx: GuideContext) -> str:
    return (
        f"√s = {ctx.sqrt_s:.3f} GeV is the total energy in the collision rest frame. "
        "Click the number to type a value, then press Enter. "
        "The window jumps to match what you type. Raising √s opens heavier finals and moves you along the curve; "
        "lowering it toward a particle threshold squeezes the available momentum and can close the channel."
    )


def _preset_body(energy: float) -> str:
    if energy < 20:
        return (
            "10 GeV sits where the Z is a small correction. A muon pair here follows the QED rule "
            "that the cross section scales as 1/s, and the tracks are soft enough to curl."
        )
    if abs(energy - M_Z) < 1:
        return (
            "The Z pole, 91.188 GeV. The resonance dominates: hadrons take about 70% of the decays, "
            "neutrinos about 20%, and each charged lepton about 3.4%. "
            "Radiation lowers the peak you see in the list."
        )
    if abs(energy - 250) < 1:
        return (
            "250 GeV is near the maximum of Born Higgsstrahlung, a few hundred femtobarns. "
            "Fermion pairs are still orders of magnitude larger, so ZH has to be selected by hand "
            "or the picture will keep showing the common channels."
        )
    return (
        "500 GeV is on the falling tail of ZH. Top pairs are open. "
        "Momenta are high, so the solenoid barely bends anything that reaches the calorimeter."
    )


def _process_body(process_id: str, ctx: GuideContext) -> str:
    rate = ""
    if process_id == ctx.process_id:
        rate = f" At this energy its rate is {format_cross_section(ctx.shown_pb)}."
    else:
        rate = " Selecting it redraws the curve and the next Collide uses this channel only."
    texts = {
        "bhabha": (
            "Electron-positron scattering through a photon, including the forward t-channel that makes the rate "
            "largest at small angles. A 10° cut keeps that pole finite. Both electrons stop in the ECAL. "
            "Z exchange is left out, so this curve has no resonance to read an asymmetry from."
        ),
        "diphoton": (
            "Annihilation to two photons. They carry no charge, so the field cannot bend them, "
            "and both stop in the ECAL. The same 10° cut is applied, and the identical-photon factor 1/2 is included."
        ),
        "zh": (
            f"Production of a real Z and a Higgs above the threshold {M_Z + M_H:.0f} GeV. "
            "Both decay before the detector. The Higgs most often gives two b quarks in the HCAL; "
            "the Z gives a lepton pair, neutrinos, or quarks. The rate peaks near 250 GeV and is far below fermion pairs."
        ),
    }
    if process_id in texts:
        return texts[process_id] + rate
    if not process_id.startswith("ff"):
        return "Final state of the hard collision." + rate
    pdg = int(process_id[2:])
    return _fermion_body(pdg) + rate


def _fermion_body(pdg: int) -> str:
    if pdg == 13:
        return (
            "A muon pair from a photon or a Z. The two tracks have opposite charge and, if they are central, "
            "cross the calorimeters without stopping and hit the muon stations. "
            "On the Z this is only a few percent of all decays, which is why it sits below the quark channels in the list."
        )
    if pdg == 15:
        return (
            "A tau pair. Each tau decays within a fraction of a millimeter (cτ ≈ 87 μm), so you never see a tau track. "
            "You see its daughters — an electron, a muon, or pions — and missing momentum from the neutrinos."
        )
    if pdg == 11:
        return (
            "An electron pair from s-channel photon and Z exchange. "
            "This is offered only when the beams are not themselves electrons; that case is Bhabha scattering. "
            "Both electrons stop in the ECAL."
        )
    if pdg in (12, 14, 16):
        return (
            f"A {species(pdg).name} {species(-pdg).name} pair. Neutrinos do not ionize, so the barrel stays empty "
            "apart from a missing-momentum arrow when the pair is not balanced in the transverse plane. "
            "On the Z, one neutrino flavor is about twice a muon pair because the Z coupling is purely left-handed and there is no photon diagram. "
            "When the beam is the same lepton flavor (ν_e from e⁻e⁺, ν_μ from μ⁻μ⁺), t-channel W exchange adds to the Z "
            "and dominates well above the Z pole."
        )
    if pdg == 6:
        return (
            "A top pair, open above about 345 GeV. Each top decays to a W and a b before it can hadronize. "
            "The W then decays, so the picture shows the b quarks and the W's daughters, not a top track."
        )
    if is_quark(pdg):
        return (
            f"A {species(pdg).name} {species(-pdg).name} pair at parton level. "
            "The rate includes three colors and a factor 1+αs/π. "
            "They are drawn into the HCAL, where a real quark would have become a jet. "
            "This laboratory does not hadronize them, so you see two lines, not a spray."
        )
    return "A fermion pair from photon and Z exchange."


def _toggle_card(field: str, ctx: GuideContext) -> GuideCard:
    if field == "isr":
        state = "on" if ctx.isr else "off"
        extra = _rate_effect(ctx)
        return GuideCard(
            f"Initial-state radiation is {state}",
            "One collinear photon is sampled from the O(β) Kuraev–Fadin radiator and the hard process runs at "
            "s′ = s(1−v). The photon itself goes down the beam pipe and is listed as γ ISR. "
            "On the Z this lowers the peak. Above the Z it can raise the rate by radiative return. "
            "Quark beams ignore the switch. "
            + extra,
        )
    if field == "force_muon":
        if ctx.force_muon:
            return GuideCard(
                "Muon decay is forced",
                "Every muon decays to an electron and two neutrinos, even though a real muon lives long enough "
                "to cross the detector (cτ = 659 m). The muon-station hit disappears, the electron stops in the ECAL, "
                "and the neutrinos add missing momentum. Turn it off to see the track a collider would record.",
            )
        return GuideCard(
            "Muons are stable",
            "Muons are kept intact and drawn out to the muon stations. "
            "That matches a real tracker: cτ is 659 m, so a muon crosses the barrel before it decays. "
            "Turn this on only when you want to see the Michel decay anyway.",
        )
    if ctx.pt_guide:
        return GuideCard(
            "pT guide is on",
            "The faint curves are not particles. They are helices for 1, 5, and 20 GeV of transverse momentum "
            "at the current field, stopped at the tracker wall (1.20 m). "
            "Use them as a ruler when the real tracks look straight. Turning the guide off removes only those curves.",
        )
    return GuideCard(
        "pT guide is off",
        "No reference helices are drawn. Turn it on to compare the event with 1, 5, and 20 GeV tracks "
        "in this magnetic field. It does not change the event.",
    )


def _solenoid_body(ctx: GuideContext) -> str:
    if ctx.b_field <= 0:
        return (
            "The field is off, so charge does not bend and every track is a straight ray. "
            "Cross sections, decays, and energies are untouched. Raise B to see curvature; "
            "it shows up first on soft tracks."
        )
    r1 = 1.0 / (P_OVER_QRB * ctx.b_field)
    r45 = 45.0 / (P_OVER_QRB * ctx.b_field)
    return (
        f"A uniform {ctx.b_field:.2f} T field along the beam, the CMS solenoid scale. "
        f"Radius follows pT = 0.2998 |q| B R, so 1 GeV bends with radius {r1:.2f} m "
        f"and 45 GeV with radius {r45:.0f} m. Positive charges curve clockwise. "
        "Changing B redraws the tracks and does not change the cross section or the decays."
    )


def _detector_body(ctx: GuideContext) -> str:
    del ctx
    return (
        "The view along the beam. Scroll the wheel to zoom. "
        "Electrons and photons stop in the blue ECAL. Quarks, gluons, and pions stop in the brown HCAL. "
        "Muons cross both and hit the outer stations. Neutrinos are dashed and leave no deposit; "
        "missing transverse momentum is drawn when it exceeds 1 GeV. "
        "The rings are a schematic barrel: each species stops where it is defined to stop, with no material shower."
    )


def _hist_body(ctx: GuideContext) -> str:
    if ctx.hist_entries <= 0:
        filled = "It is empty until you collide."
    elif ctx.hist_entries == 1:
        filled = "It holds the one event on screen."
    else:
        filled = f"It holds {ctx.hist_entries} events at the current settings."
    return (
        "Each entry is √s′, the energy of the hard collision after any initial-state photon. "
        "With radiation off, every entry sits on the beam energy. "
        "With radiation on, the peak spreads, and above the Z some entries pile up back at 91 GeV. "
        + filled
        + " Clear empties it. Changing the beam or jumping far in energy starts a new window."
    )


def _shape_body(ctx: GuideContext) -> str:
    return (
        "The blue-gray curve is the Born cross section of the selected process. Gold includes the O(β) leading-log radiator "
        "when the beams are leptons and ISR is on; otherwise the two curves coincide. "
        f"The cyan line is your setting, {ctx.sqrt_s:.3f} GeV. "
        "On the Z the gold peak is lower than the blue one. Above the Z the gold curve can sit higher, "
        "which is radiative return. The marker does not have to be on a peak: that is the rate at the energy you chose."
    )


def _conservation_body() -> str:
    return (
        "The generator refuses an event that breaks a conservation law. "
        "ΔE and Δp are the leftover energy and momentum; they should be numerical dust, around 10⁻¹⁴ GeV. "
        "ΔQ is electric charge in units of the proton charge. "
        "ΔL is three integers, (electron, muon, tau) flavor, so a muon cannot turn into an electron "
        "unless the neutrinos balance it. Baryon number is checked the same way. "
        "Green means this event passed all of them."
    )


def theta_degrees(cos_theta: float) -> float:
    cosine = min(1.0, max(-1.0, cos_theta))
    return math.degrees(math.acos(cosine))
