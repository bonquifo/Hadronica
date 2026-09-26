"""Display of full PYTHIA events: hundreds of tracks from their true vertices.

Each drawable particle becomes a :class:`Trace` in meters. Final particles run
from their production vertex to the layer where that species stops; decayed
particles (K0_S, Λ, b and c hadrons, τ, …) run to their actual decay vertex,
so displaced vertices appear where PYTHIA put them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from hadronica.scene import ECAL, ECAL_HALF, HCAL, HCAL_HALF, MUONS, MUON_HALF, OUTER_M, TRACKER_HALF, TRACKER_RADII, Camera, _sector
from hadronica.theme import GOLD, mix
from hadronica.tracks import path_from_vertex

# Display thresholds (a real event display applies similar cuts).
MIN_PT_CHARGED = 0.3  # GeV
MIN_E_NEUTRAL = 0.5  # GeV
N_TOWERS = 72  # 5° in φ, the CMS HCAL tower width

NEUTRINOS = (12, 14, 16)
PARTON_IDS = set(range(1, 9)) | {21, 90} | {1103, 2101, 2103, 2203, 3101, 3103, 3201, 3203, 3303, 4101, 4103, 4201, 4203,
                                         4301, 4303, 4403, 5101, 5103, 5201, 5203, 5301, 5303, 5401, 5403, 5503}


@dataclass(slots=True)
class Trace:
    points: list[tuple[float, float, float]]
    color: tuple[int, int, int]
    charged: bool
    kind: str  # "track", "photon", "neutral", "decayed"
    label: str | None
    deposit: str | None  # "ecal", "hcal", "muon" or None
    energy: float
    phi_end: float


def _class_color(pdg: int, charge: float) -> tuple[int, int, int]:
    a = abs(pdg)
    if a == 11:
        return (88, 214, 255)
    if a == 13:
        return (150, 160, 255)
    if a == 22:
        return (255, 226, 120)
    if a == 15:
        return (196, 146, 255)
    if a in (310, 3122, 3112, 3222, 3312, 3334):
        return (120, 240, 170)  # strange V0s and hyperons
    if 400 <= a < 600 or 4000 <= a < 6000:
        return (255, 120, 170)  # charm and bottom hadrons
    if abs(charge) > 1.0e-6:
        return (255, 178, 102)  # charged hadrons
    return (170, 178, 196)  # neutral hadrons


def build_traces(event, b_field: float) -> list[Trace]:
    traces: list[Trace] = []
    for particle in event.particles:
        a = abs(particle.pdg)
        if a in PARTON_IDS or a in NEUTRINOS or abs(particle.status) in (11, 12):
            continue
        x, y, z = (c * 1.0e-3 for c in particle.vertex_mm)
        charged = abs(particle.charge) > 1.0e-6
        pt = particle.p4.pt
        if particle.final:
            if charged and pt < MIN_PT_CHARGED:
                continue
            if not charged and particle.p4.e < MIN_E_NEUTRAL:
                continue
            if a in (11, 22):
                r_stop, z_stop, deposit = ECAL[0], ECAL_HALF, "ecal"
            elif a == 13:
                r_stop, z_stop, deposit = MUONS[-1][1], MUON_HALF, "muon"
            else:
                r_stop, z_stop, deposit = HCAL[0], HCAL_HALF, "hcal"
            max_length = None
            kind = "photon" if a == 22 else ("track" if charged else "neutral")
        else:
            # Decayed in flight: draw only if the decay vertex is visibly displaced.
            decay = event.decay_vertex_mm(particle)
            if decay is None or (abs(particle.status) < 81 and a != 15):
                continue
            flight_mm = math.dist(decay, particle.vertex_mm)
            if flight_mm < 0.5:
                continue
            r_stop, z_stop, deposit = HCAL[0], HCAL_HALF, None
            max_length = flight_mm * 1.0e-3
            # A neutral parent leaves no hits: it is drawn dashed, as event displays show V0s.
            kind = "decayed" if charged else "decayed_neutral"
            if pt < MIN_PT_CHARGED:
                continue
        points = path_from_vertex(
            (x, y, z),
            particle.p4.px,
            particle.p4.py,
            particle.p4.pz,
            particle.charge if charged else 0.0,
            b_field,
            r_stop,
            z_stop,
            max_length=max_length,
        )
        if len(points) < 2:
            continue
        end = points[-1]
        reached = math.hypot(end[0], end[1]) >= r_stop * 0.995
        label = None
        if particle.final and a in (11, 13) and pt > 10.0:
            label = "e" if a == 11 else "μ"
        if particle.final and a == 22 and pt > 20.0:
            label = "γ"
        traces.append(
            Trace(
                points=points,
                color=_class_color(particle.pdg, particle.charge),
                charged=charged,
                kind=kind,
                label=label,
                deposit=deposit if reached else None,
                energy=particle.p4.e,
                phi_end=math.atan2(end[1], end[0]),
            )
        )
    traces.extend(_pileup_traces(event, b_field))
    return traces


PILEUP_COLOR = (92, 104, 128)


def _pileup_traces(event, b_field: float) -> list[Trace]:
    """Tracks Delphes flags as pileup, from their own vertices along the beam, to the tracker wall."""
    reco = getattr(event, "reco", None)
    if not reco:
        return []
    out = []
    for pt, eta, phi, charge, z_mm, is_pu in reco.get("tracks_z", []):
        if not is_pu or pt < 0.5:
            continue
        px, py, pz = pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)
        points = path_from_vertex((0.0, 0.0, z_mm * 1.0e-3), px, py, pz, charge, b_field,
                                  TRACKER_RADII[-1], TRACKER_HALF, step_m=0.08)
        if len(points) > 1:
            end = points[-1]
            out.append(Trace(points, PILEUP_COLOR, True, "pileup", None, None, 0.0, math.atan2(end[1], end[0])))
    return out


def tower_sums(traces: list[Trace]) -> dict[str, list[float]]:
    towers = {"ecal": [0.0] * N_TOWERS, "hcal": [0.0] * N_TOWERS}
    width = 2.0 * math.pi / N_TOWERS
    for trace in traces:
        if trace.deposit in towers:
            index = int(((trace.phi_end + math.pi) % (2.0 * math.pi)) / width) % N_TOWERS
            towers[trace.deposit][index] += trace.energy
    return towers


def _jet_list(event) -> list[tuple[float, float, bool]]:
    """(pT, φ, b-tagged) for the jets to draw: Delphes jets when simulated, else truth jets."""
    reco = getattr(event, "reco", None)
    if reco and "jets" in reco:
        return [(j[0], j[2], bool(int(j[4]) & 1)) for j in reco["jets"]]
    return [(jet.pt, math.atan2(jet.py, jet.px), False) for jet in event.jets if jet.pt > 0.0]


def reco_towers(reco: dict) -> dict[str, list[float]]:
    """Delphes calorimeter towers (E_em, E_had) summed into the display's φ bins."""
    towers = {"ecal": [0.0] * N_TOWERS, "hcal": [0.0] * N_TOWERS}
    width = 2.0 * math.pi / N_TOWERS
    for _et, _eta, phi, _e, eem, ehad in reco.get("towers", []):
        index = int(((phi + math.pi) % (2.0 * math.pi)) / width) % N_TOWERS
        towers["ecal"][index] += eem
        towers["hcal"][index] += ehad
    return towers


def draw_full_event(surf, camera: Camera, event, traces: list[Trace], fraction: float, font, report) -> None:
    fraction = max(0.0, min(1.0, fraction))
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    offset = (0, 0)
    reco = getattr(event, "reco", None)
    # Jets: translucent wedges spanning the jet radius in φ; b-tagged jets in pink.
    for index, (_pt, phi, btag) in enumerate(_jet_list(event)[:8]):
        half = 0.4 if event.beam_id == "pp" else 0.3
        poly = [camera.to_screen(0.0, 0.0)]
        for i in range(9):
            angle = phi - half + 2 * half * i / 8
            poly.append(camera.to_screen(HCAL[1] * math.cos(angle), HCAL[1] * math.sin(angle)))
        tint = (255, 120, 170) if btag else (255, 206, 120)
        pygame.draw.polygon(layer, (*tint, int(26 * fraction)), poly)
        pygame.draw.lines(layer, (*tint, int(90 * fraction)), True, poly, 1)
    # Calorimeter towers, bar length ∝ log(1 + E).
    if fraction > 0.5:
        towers = reco_towers(reco) if reco and "towers" in reco else tower_sums(traces)
        width = 2.0 * math.pi / N_TOWERS
        for name, (r0, r1), color in (("ecal", ECAL, (90, 190, 255)), ("hcal", HCAL, (255, 170, 90))):
            values = towers[name]
            top = max(max(values), 1.0)
            for index, energy in enumerate(values):
                if energy < 0.3:
                    continue
                length = (r1 - r0) * min(1.0, math.log10(1.0 + energy) / math.log10(1.0 + top))
                phi0 = -math.pi + index * width
                poly = _sector(camera, r0, r0 + max(0.04, length), phi0, phi0 + width)
                pygame.draw.polygon(layer, (*color, int(210 * min(1.0, (fraction - 0.5) * 2.0))), poly)
    surf.blit(layer, offset)

    # Tracks.
    labels = []
    for trace in traces:
        count = max(2, int(1 + fraction * (len(trace.points) - 1)))
        pts = [camera.to_screen(p[0], p[1]) for p in trace.points[:count]]
        if len(pts) < 2:
            continue
        if trace.kind in ("photon", "decayed_neutral"):
            if trace.kind == "decayed_neutral" and count == len(trace.points):
                end = pts[-1]
                pygame.draw.circle(surf, trace.color, (int(end[0]), int(end[1])), 3, 1)
            for i in range(0, len(pts) - 1, 2):
                pygame.draw.line(surf, trace.color, pts[i], pts[i + 1], 1)
        elif trace.kind == "pileup":
            pygame.draw.lines(surf, trace.color, False, pts, 1)
        elif trace.kind == "neutral":
            pygame.draw.lines(surf, mix(trace.color, (8, 11, 19), 0.35), False, pts, 1)
        elif trace.kind == "decayed":
            pygame.draw.lines(surf, trace.color, False, pts, 2)
            if count == len(trace.points):
                end = pts[-1]
                pygame.draw.circle(surf, trace.color, (int(end[0]), int(end[1])), 3, 1)
        else:
            width = 2 if trace.label else 1
            pygame.draw.lines(surf, trace.color, False, pts, width)
        if trace.label and fraction > 0.92:
            labels.append((pts[-1], trace.label, trace.color))
    for (x, y), text, color in labels:
        image = font.render(text, True, color)
        surf.blit(image, (x + 6, y - 14))
    # Jet labels.
    if fraction > 0.92:
        for pt, phi, btag in _jet_list(event)[:6]:
            at = camera.to_screen((HCAL[1] + 0.25) * math.cos(phi), (HCAL[1] + 0.25) * math.sin(phi))
            text = f"b-jet {pt:.0f} GeV" if btag else f"jet {pt:.0f} GeV"
            image = font.render(text, True, (255, 120, 170) if btag else (255, 206, 120))
            surf.blit(image, image.get_rect(center=at))
    # Reconstructed leptons and photons, marked where they enter the calorimeter.
    if reco and fraction > 0.92:
        marks = []
        for pt, _eta, phi, charge in reco.get("electrons", []):
            marks.append((phi, ECAL[0], f"e{'⁺' if charge > 0 else '⁻'} {pt:.0f}", (88, 214, 255)))
        for pt, _eta, phi, charge in reco.get("muons", []):
            marks.append((phi, MUONS[-1][1], f"μ{'⁺' if charge > 0 else '⁻'} {pt:.0f}", (150, 160, 255)))
        for pt, _eta, phi, _e in reco.get("photons", []):
            if pt > 5.0:
                marks.append((phi, ECAL[0], f"γ {pt:.0f}", (255, 226, 120)))
        for phi, radius, text, color in marks:
            at = camera.to_screen(radius * math.cos(phi), radius * math.sin(phi))
            pygame.draw.circle(surf, color, (int(at[0]), int(at[1])), 6, 2)
            image = font.render(text + " GeV", True, color)
            surf.blit(image, (at[0] + 9, at[1] - 9))
    # Missing transverse momentum: reconstructed when simulated, else from the truth record.
    met_vector = None
    if reco and "met" in reco:
        value, phi = reco["met"]
        met_vector = (value, value * math.cos(phi), value * math.sin(phi))
    elif report is not None:
        met_vector = (math.hypot(report.missing_px, report.missing_py), report.missing_px, report.missing_py)
    if met_vector is not None and fraction > 0.85:
        met, mpx, mpy = met_vector
        if met > 5.0:
            length = min(ECAL[0], 0.02 * met + 0.3)
            ux, uy = mpx / met, mpy / met
            start = camera.origin
            end = camera.to_screen(ux * length, uy * length)
            pygame.draw.line(surf, (255, 110, 150), start, end, 3)
            image = font.render(f"missing pT {met:.1f} GeV", True, (255, 110, 150))
            surf.blit(image, (end[0] + 6, end[1] - 8))
    pygame.draw.circle(surf, GOLD, (int(camera.origin[0]), int(camera.origin[1])), 3)


def count_summary(event) -> dict[str, int]:
    """Multiplicities of final particles by class."""
    counts = {"charged": 0, "photons": 0, "neutral_hadrons": 0, "leptons": 0, "neutrinos": 0}
    for particle in event.finals():
        a = abs(particle.pdg)
        if a in NEUTRINOS:
            counts["neutrinos"] += 1
        elif a in (11, 13):
            counts["leptons"] += 1
            counts["charged"] += 1
        elif a == 22:
            counts["photons"] += 1
        elif abs(particle.charge) > 1.0e-6:
            counts["charged"] += 1
        else:
            counts["neutral_hadrons"] += 1
    return counts


__all__ = [
    "Trace",
    "build_traces",
    "count_summary",
    "draw_full_event",
    "tower_sums",
    "TRACKER_HALF",
    "TRACKER_RADII",
    "OUTER_M",
]
