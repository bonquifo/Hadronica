"""Delphes fast detector simulation for PYTHIA events (runs inside WSL).

PYTHIA's event record is converted to HepMC3 with the full vertex structure
(one vertex per decay or branching, positioned at the daughters' production
point), passed through ``DelphesHepMC3`` with the detector card that matches
the collider, and the reconstructed objects are read back with uproot.

Cards (shipped with Delphes 3.5):
    p p                 delphes_card_CMS.tcl           (CMS, 3.8 T)
    e+e- ≤ 209 GeV      delphes_card_ALEPH.tcl         (LEP, ALEPH)
    e+e- ≤ 400 GeV      delphes_card_IDEA.tcl          (FCC-ee, IDEA)
    e+e- ≤ 1.5 TeV      delphes_card_CLICdet_Stage2.tcl
    e+e- > 1.5 TeV      delphes_card_CLICdet_Stage3.tcl
    μ+μ-                delphes_card_MuonColliderDet.tcl
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import pyhepmc as hep

CARD_DIR = os.path.join(sys.prefix, "cards")
DELPHES = os.path.join(sys.prefix, "bin", "DelphesHepMC3")


def card_for(beams: str, sqrt_s: float) -> tuple[str, str]:
    """Return (card path, human-readable detector name)."""
    if beams == "pp":
        name, label = "delphes_card_CMS.tcl", "CMS (Delphes)"
    elif beams == "mumu":
        name, label = "delphes_card_MuonColliderDet.tcl", "Muon Collider detector (Delphes)"
    elif sqrt_s <= 209.0:
        name, label = "delphes_card_ALEPH.tcl", "ALEPH at LEP (Delphes)"
    elif sqrt_s <= 400.0:
        name, label = "delphes_card_IDEA.tcl", "IDEA at FCC-ee (Delphes)"
    elif sqrt_s <= 1500.0:
        name, label = "delphes_card_CLICdet_Stage2.tcl", "CLICdet (Delphes)"
    else:
        name, label = "delphes_card_CLICdet_Stage3.tcl", "CLICdet (Delphes)"
    return os.path.join(CARD_DIR, name), label


CACHE_DIR = os.path.join(os.path.expanduser("~"), "hadronica-cache")
PILEUP_LIBRARY_EVENTS = 5000


def pileup_library(sqrt_s: float) -> str:
    """Minimum-bias events for pileup, in Delphes' binary format; built once per energy.

    PYTHIA SoftQCD:inelastic with the Monash tune and its standard decays,
    converted with Delphes' hepmc2pileup. PileUpMerger draws from this file
    at random for every simulated bunch crossing.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"minbias_{int(round(sqrt_s))}GeV_{PILEUP_LIBRARY_EVENTS}.pileup")
    if os.path.exists(path):
        return path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from worker import Worker

    worker = Worker()
    worker.init({"beams": "pp", "process": "pp_minbias", "sqrt_s": sqrt_s, "seed": 20260923, "decays": "generator"})
    with tempfile.TemporaryDirectory(prefix="hadronica-minbias-") as work:
        source = os.path.join(work, "minbias.hepmc2")
        with hep.io.WriterAsciiHepMC2(source) as writer:
            done = 0
            while done < PILEUP_LIBRARY_EVENTS:
                for raw in worker.generate(min(500, PILEUP_LIBRARY_EVENTS - done), jets=False)["events"]:
                    writer.write(to_hepmc(raw["particles"], done))
                    done += 1
        partial = path + ".part"
        subprocess.run([os.path.join(sys.prefix, "bin", "hepmc2pileup"), partial, source], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.replace(partial, path)
    return path


def pileup_card(mean_pileup: float, library: str, work: str) -> str:
    """The Delphes CMS pileup card with this ⟨μ⟩ and library, writing tracks and towers."""
    with open(os.path.join(CARD_DIR, "delphes_card_CMS_PileUp.tcl"), encoding="utf-8") as handle:
        text = handle.read()
    replacements = {
        "set PileUpFile MinBias.pileup": f"set PileUpFile {library}",
        "set MeanPileUp 50": f"set MeanPileUp {mean_pileup:g}",
        "#  add Branch TrackMerger/tracks Track Track": "  add Branch TrackMerger/tracks Track Track",
        # This card runs separate ECAL and HCAL tower lists (its combined merger is
        # not on the execution path); write both and combine them when reading.
        "#  add Branch Calorimeter/towers Tower Tower":
            "  add Branch ECal/ecalTowers ECalTower Tower\n  add Branch HCal/hcalTowers HCalTower Tower",
        "  add Branch Delphes/allParticles Particle GenParticle": "#  add Branch Delphes/allParticles Particle GenParticle",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"unexpected CMS pileup card: missing '{old}'")
        text = text.replace(old, new)
    path = os.path.join(work, "delphes_card_CMS_PileUp_hadronica.tcl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _status_hepmc(pdg: int, status: int, has_daughters: bool) -> int:
    """HepMC status convention (1 final, 2 decayed physical particle, 4 beam, else |status|)."""
    if status == -12:
        return 4
    if status > 0:
        return 1
    a = abs(pdg)
    physical = a > 100 or a in (11, 13, 15, 22)
    if physical and has_daughters and (abs(status) >= 81 or a == 15):
        return 2
    return abs(status)


def to_hepmc(rows: list[list], number: int) -> "hep.GenEvent":
    """Build a HepMC3 event from serialized PYTHIA rows (see worker._serialize)."""
    event = hep.GenEvent(hep.Units.GEV, hep.Units.MM)
    event.event_number = number
    particles = [None]
    for index in range(1, len(rows)):
        pdg, status, _m1, _m2, d1, d2, px, py, pz, e, m, *_rest = rows[index]
        particle = hep.GenParticle((px, py, pz, e), int(pdg), _status_hepmc(int(pdg), int(status), d1 > 0))
        particle.generated_mass = m
        particles.append(particle)
    vertices: dict[tuple, "hep.GenVertex"] = {}
    incoming: dict[tuple[int, ...], list[int]] = {}
    made_by: dict[int, tuple[int, ...]] = {}
    for index in range(1, len(rows)):
        row = rows[index]
        d1, d2 = int(row[4]), int(row[5])
        if d1 <= 0:
            continue
        if d2 >= d1:
            kids = tuple(range(d1, d2 + 1))
        elif d2 > 0:
            kids = (d1, d2)
        else:
            kids = (d1,)
        kids = tuple(k for k in kids if 0 < k < len(rows))
        if not kids:
            continue
        if kids not in vertices:
            first = rows[kids[0]]
            vertices[kids] = hep.GenVertex((first[11], first[12], first[13], 0.0))
            incoming[kids] = []
            for kid in kids:
                if kid not in made_by:
                    vertices[kids].add_particle_out(particles[kid])
                    made_by[kid] = kids
        vertices[kids].add_particle_in(particles[index])
        incoming[kids].append(index)
    # Some particles (the QED beam remnants of a lepton beam, for example) are
    # not in their mother's daughter range. Attach each to its mother's end vertex.
    end_of = {index: key for key, parents in incoming.items() for index in parents}
    for index in range(1, len(rows)):
        if index in made_by or int(rows[index][1]) == -12:
            continue
        mother = int(rows[index][2])
        if mother <= 0 or mother >= len(rows):
            continue
        key = end_of.get(mother)
        if key is None:
            key = ("remnant", mother)
            row = rows[index]
            vertices[key] = hep.GenVertex((row[11], row[12], row[13], 0.0))
            vertices[key].add_particle_in(particles[mother])
            incoming[key] = [mother]
            end_of[mother] = key
        vertices[key].add_particle_out(particles[index])
        made_by[index] = key
    # PYTHIA's backward initial-state shower lists particles before their parents;
    # HepMC3 needs every vertex added after the vertices that make its incoming particles.
    order: list[tuple[int, ...]] = []
    state: dict[tuple[int, ...], int] = {}

    def visit(key: tuple[int, ...]) -> None:
        stack = [(key, iter(incoming[key]))]
        state[key] = 1
        while stack:
            current, parents = stack[-1]
            advanced = False
            for parent in parents:
                upstream = made_by.get(parent)
                if upstream is not None and upstream != current and state.get(upstream, 0) == 0:
                    state[upstream] = 1
                    stack.append((upstream, iter(incoming[upstream])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                state[current] = 2
                order.append(current)

    for key in vertices:
        if state.get(key, 0) == 0:
            visit(key)
    for key in order:
        event.add_vertex(vertices[key])
    return event


def _read(tree, branch: str, fields: tuple[str, ...]):
    names = tree.keys()
    wanted = [f"{branch}/{branch}.{field}" for field in fields]
    if not all(name in names for name in wanted):
        return None
    arrays = tree.arrays(wanted, library="np")
    return [arrays[name] for name in wanted]


def simulate(events: list[dict], beams: str, sqrt_s: float, pileup: float = 0.0) -> tuple[list[dict], str]:
    """Run Delphes on serialized events; return one reconstruction dict per event.

    ``pileup`` > 0 (proton beams only) overlays a Poisson number of minimum-bias
    collisions with that mean, spread along the beam as in the CMS card.
    """
    import uproot

    card, label = card_for(beams, sqrt_s)
    with tempfile.TemporaryDirectory(prefix="hadronica-delphes-") as work:
        if beams == "pp" and pileup > 0.0:
            card = pileup_card(pileup, pileup_library(sqrt_s), work)
            label = f"CMS (Delphes), mean pileup μ = {pileup:g}"
        hepmc = os.path.join(work, "events.hepmc")
        root = os.path.join(work, "delphes.root")
        with hep.io.WriterAscii(hepmc) as writer:
            for number, raw in enumerate(events):
                writer.write(to_hepmc(raw["particles"], number))
        result = subprocess.run(
            [DELPHES, card, root, hepmc],
            cwd=CARD_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "ROOT_INCLUDE_PATH": os.path.join(sys.prefix, "include")},
        )
        if result.returncode != 0 or not os.path.exists(root):
            tail = "\n".join(result.stdout.strip().splitlines()[-6:])
            raise RuntimeError(f"Delphes failed ({os.path.basename(card)}): {tail}")
        tree = uproot.open(root)["Delphes"]
        out = [dict() for _ in events]
        specs = {
            "electrons": ("Electron", ("PT", "Eta", "Phi", "Charge")),
            "muons": ("Muon", ("PT", "Eta", "Phi", "Charge")),
            "photons": ("Photon", ("PT", "Eta", "Phi", "E")),
            "towers": ("Tower", ("ET", "Eta", "Phi", "E", "Eem", "Ehad")),
            "tracks": ("Track", ("PT", "Eta", "Phi", "Charge")),
        }
        for key, (branch, fields) in specs.items():
            columns = _read(tree, branch, fields)
            if columns is None:
                continue
            for i in range(len(events)):
                out[i][key] = [
                    [round(float(col[i][j]), 5) for col in columns] for j in range(len(columns[0][i]))
                ]
        # Jets: "Jet" in the CMS, ALEPH, and IDEA cards; the CLICdet and muon-collider
        # cards provide VLC jets, of which R = 0.7 inclusive (energy-smeared when
        # available) is used.
        for branch in ("Jet", "JER_VLCjetR07_inclusive", "VLCjetR07_inclusive", "VLCjetR05_inclusive"):
            columns = _read(tree, branch, ("PT", "Eta", "Phi", "Mass", "BTag", "TauTag"))
            if columns is None:
                continue
            for i in range(len(events)):
                out[i]["jets"] = [
                    [round(float(col[i][j]), 5) for col in columns] for j in range(len(columns[0][i]))
                ]
                out[i]["jet_collection"] = branch
            break
        if all("towers" not in rec for rec in out):
            ecal = _read(tree, "ECalTower", ("ET", "Eta", "Phi", "E"))
            hcal = _read(tree, "HCalTower", ("ET", "Eta", "Phi", "E"))
            if ecal is not None and hcal is not None:
                for i in range(len(events)):
                    towers = [[float(ecal[0][i][j]), float(ecal[1][i][j]), float(ecal[2][i][j]), float(ecal[3][i][j]),
                               float(ecal[3][i][j]), 0.0] for j in range(len(ecal[0][i]))]
                    towers += [[float(hcal[0][i][j]), float(hcal[1][i][j]), float(hcal[2][i][j]), float(hcal[3][i][j]),
                                0.0, float(hcal[3][i][j])] for j in range(len(hcal[0][i]))]
                    out[i]["towers"] = [[round(v, 5) for v in tower] for tower in towers]
        # Reconstructed tracks with their production z and pileup flag, for the display.
        track_fields = ("PT", "Eta", "Phi", "Charge", "Z", "IsPU")
        columns = _read(tree, "Track", track_fields)
        if columns is not None:
            for i in range(len(events)):
                out[i]["tracks_z"] = [
                    [round(float(col[i][j]), 4) for col in columns] for j in range(len(columns[0][i]))
                ]
        vertices = _read(tree, "Vertex", ("Z",))
        if vertices is not None:
            for i in range(len(events)):
                out[i]["n_vertices"] = int(len(vertices[0][i]))
                out[i]["vertex_z"] = [round(float(z), 3) for z in vertices[0][i][:400]]
        met = _read(tree, "MissingET", ("MET", "Phi"))
        if met is not None:
            for i in range(len(events)):
                if len(met[0][i]):
                    out[i]["met"] = [float(met[0][i][0]), float(met[1][i][0])]
        for rec in out:
            rec["detector"] = label
        return out, label


def _selftest() -> None:
    """Delphes on a few PYTHIA events: python hep/detector.py"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from worker import Worker

    for beams, process, energy in (("pp", "pp_ttbar", 13600.0), ("ee", "ll_gmz_had", 91.1879), ("ee", "ll_zh", 240.0)):
        worker = Worker()
        worker.init({"beams": beams, "process": process, "sqrt_s": energy, "seed": 3})
        events = worker.generate(5)["events"]
        recos, label = simulate(events, beams, energy)
        for raw, reco in zip(events[:2], recos[:2]):
            jets = reco.get("jets", [])
            print(
                f"{label:30s} {process:11s} truth jets {len(raw['jets'])}  reco jets {len(jets)}"
                f" (b-tagged {sum(1 for j in jets if j[4] > 0)})  e {len(reco.get('electrons', []))}"
                f"  μ {len(reco.get('muons', []))}  γ {len(reco.get('photons', []))}"
                f"  tracks {len(reco.get('tracks', []))}  towers {len(reco.get('towers', []))}"
                f"  MET {reco.get('met', [math.nan])[0]:.1f}"
            )


if __name__ == "__main__":
    _selftest()
