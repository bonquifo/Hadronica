"""PYTHIA 8 event worker for Hadronica. Runs inside WSL (Linux) and talks JSON lines.

Protocol, one JSON object per line on stdin; one JSON reply per line on stdout:

    {"cmd": "hello"}                              -> {"ok": true, "pythia": "8.3xx", ...}
    {"cmd": "init", "config": {...}}              -> {"ok": true, "sigma_pb": ..., "log": [...]}
    {"cmd": "generate", "n": 1}                   -> {"ok": true, "events": [...], "sigma_pb": ..., ...}
    {"cmd": "quit"}

Physics comes entirely from PYTHIA 8.3 (hard process, QED/QCD initial- and
final-state showers, multiparton interactions, Lund string hadronization,
hadron and tau decays) with its default Monash 2013 tune. The only changes to
PYTHIA's defaults are electroweak inputs set to PDG 2026 values (see
``PDG_OVERRIDES``) and decays restricted to the tracker volume, so long-lived
particles decay where a real detector would see them.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fastjet_cxx as fastjet  # noqa: E402
import pythia8  # noqa: E402

# The conda activation script normally sets PYTHIA8DATA; the worker runs without activation.
XMLDIR = os.environ.get("PYTHIA8DATA") or os.path.join(sys.prefix, "share", "Pythia8", "xmldoc")

# PDG 2026 (Int. J. Mod. Phys. A 41, 2630011) masses and widths, in GeV.
# Couplings and the tune's α_s are left at PYTHIA's tuned defaults.
PDG_OVERRIDES = (
    "23:m0 = 91.1879",
    "23:mWidth = 2.4955",
    "24:m0 = 80.3625",
    "24:mWidth = 2.14",
    "25:m0 = 125.13",
    "25:mWidth = 0.004101",  # SM width, LHC Higgs XS WG at 125.10 GeV
    "6:m0 = 172.60",
    "6:mWidth = 1.42",
    "15:m0 = 1.77693",
    "StandardModel:sin2thetaWbar = 0.23154",
    # PYTHIA otherwise recomputes resonance widths from its own partial widths at
    # init (W: 2.092 GeV, t: 1.347 GeV); keep the PDG totals, rescaling the partial
    # widths so the branching ratios are unchanged. PYTHIA ignores this for the Z:
    # its γ*/Z treatment computes the width from the couplings (2.504 GeV, 0.34 %
    # above PDG; the Z-pole hadronic cross section is 41.46 nb, measured 41.48 nb).
    "23:doForceWidth = on",
    "24:doForceWidth = on",
    "25:doForceWidth = on",
    "6:doForceWidth = on",
)

LEPTON_BEAMS = {"ee": (11, -11), "mumu": (13, -13)}

# Each process: (title, settings, extra settings for pp only).
PROCESSES: dict[str, dict] = {
    # ---- lepton colliders -------------------------------------------------
    "ll_gmz_all": {
        "beams": "lepton",
        "title": "γ*/Z → f f̄ (all flavors)",
        "settings": ["WeakSingleBoson:ffbar2gmZ = on"],
    },
    "ll_gmz_had": {
        "beams": "lepton",
        "title": "γ*/Z → hadrons",
        "settings": ["WeakSingleBoson:ffbar2gmZ = on", "23:onMode = off", "23:onIfAny = 1 2 3 4 5"],
    },
    "ll_gmz_mu": {
        "beams": "lepton",
        "title": "γ*/Z → μ⁺μ⁻",
        "settings": ["WeakSingleBoson:ffbar2gmZ = on", "23:onMode = off", "23:onIfAny = 13"],
    },
    "ll_gmz_tau": {
        "beams": "lepton",
        "title": "γ*/Z → τ⁺τ⁻",
        "settings": ["WeakSingleBoson:ffbar2gmZ = on", "23:onMode = off", "23:onIfAny = 15"],
    },
    "ll_ww": {
        "beams": "lepton",
        "title": "W⁺W⁻",
        "settings": ["WeakDoubleBoson:ffbar2WW = on"],
    },
    "ll_zz": {
        "beams": "lepton",
        "title": "Z Z",
        "settings": ["WeakDoubleBoson:ffbar2gmZgmZ = on", "PhaseSpace:mHatMin = 20.",
                     # Pure Z Z: PYTHIA's default γ*/Z mixture adds γ*γ* and γ*Z pairs (1.35 pb
                     # instead of 1.00 pb at 200 GeV; LEP measured about 1.0 pb).
                     "WeakZ0:gmZmode = 2"],
    },
    "ll_zh": {
        "beams": "lepton",
        "title": "Z H (Higgsstrahlung)",
        "settings": ["HiggsSM:ffbar2HZ = on"],
    },
    "ll_vbf_h": {
        "beams": "lepton",
        "title": "ν ν̄ H (WW fusion)",
        "settings": ["HiggsSM:ff2Hff(t:WW) = on"],
    },
    "ll_ttbar": {
        "beams": "lepton",
        "title": "t t̄",
        "settings": ["Top:ffbar2ttbar(s:gmZ) = on"],
    },
    # ---- proton-proton ----------------------------------------------------
    "pp_z_ll": {
        "beams": "pp",
        "title": "Drell–Yan Z/γ* → ℓ⁺ℓ⁻",
        "settings": [
            "WeakSingleBoson:ffbar2gmZ = on", "23:onMode = off", "23:onIfAny = 11 13",
            "PhaseSpace:mHatMin = 60.",
        ],
    },
    "pp_w_lnu": {
        "beams": "pp",
        "title": "W → ℓν",
        "settings": ["WeakSingleBoson:ffbar2W = on", "24:onMode = off", "24:onIfAny = 11 13"],
    },
    "pp_ttbar": {
        "beams": "pp",
        "title": "t t̄",
        "settings": ["Top:gg2ttbar = on", "Top:qqbar2ttbar = on"],
    },
    "pp_h_gg": {
        "beams": "pp",
        "title": "gg → H → γγ",
        "settings": ["HiggsSM:gg2H = on", "25:onMode = off", "25:onIfMatch = 22 22"],
    },
    "pp_h_4l": {
        "beams": "pp",
        "title": "gg → H → ZZ* → 4ℓ",
        "settings": [
            "HiggsSM:gg2H = on", "25:onMode = off", "25:onIfMatch = 23 23",
            "23:onMode = off", "23:onIfAny = 11 13",
        ],
    },
    "pp_h_all": {
        "beams": "pp",
        "title": "Higgs, all production and decay modes",
        "settings": ["HiggsSM:all = on"],
    },
    "pp_vv": {
        "beams": "pp",
        "title": "Dibosons WW, WZ, ZZ",
        "settings": ["WeakDoubleBoson:all = on", "WeakZ0:gmZmode = 2"],
    },
    "pp_dijet": {
        "beams": "pp",
        "title": "QCD dijets, p̂T > 100 GeV",
        "settings": ["HardQCD:all = on", "PhaseSpace:pTHatMin = 100."],
    },
    "pp_minbias": {
        "beams": "pp",
        "title": "Minimum bias (inelastic)",
        "settings": ["SoftQCD:inelastic = on"],
    },
}

MB_TO_PB = 1.0e9

NLO_DIR = os.path.join(os.path.expanduser("~"), "smlab-cache", "nlo")
# Hadronica process → MadGraph5_aMC@NLO sample names, most accurate first
# (see hep/mg5/generate_nlo.sh): t t̄ with MadSpin spin-correlated decays, and
# FxFx-merged Z + 0, 1, 2 jets at NLO.
NLO_SAMPLES = {"pp_ttbar": ("ttbar_ms", "ttbar"), "pp_z_ll": ("dy_fxfx", "dy"), "pp_w_lnu": ("w",)}

# PYTHIA 8 settings for showering MC@NLO events, exactly as MadGraph5_aMC@NLO
# writes them (Template/NLO/MCatNLO/Scripts/MCatNLO_MadFKS_PYTHIA8.Script):
# shower starting scales from the LHE file, global recoil for the first FSR
# emission, no matrix-element corrections (the NLO calculation supplies them),
# and first-order α_s(M_Z) = 0.118 in both showers.
MCATNLO_SETTINGS = (
    "Beams:frameType = 4",
    "Beams:setProductionScalesFromLHEF = on",
    "Check:epTolErr = 0.001",
    "TimeShower:pTmaxMatch = 1",
    "TimeShower:pTmaxFudge = 1.",
    "TimeShower:alphaSvalue = 0.118",
    "TimeShower:alphaSorder = 1",
    "TimeShower:alphaEMorder = 0",
    "TimeShower:dampenBeamRecoil = off",
    "TimeShower:globalRecoil = on",
    "TimeShower:nMaxGlobalRecoil = 1",
    "TimeShower:globalRecoilMode = 2",
    "TimeShower:nMaxGlobalBranch = 1",
    "TimeShower:nPartonsInBorn = -1",
    "TimeShower:limitPTmaxGlobal = on",
    "TimeShower:alphaSuseCMW = false",
    "TimeShower:weightGluonToQuark = 1",
    "SpaceShower:pTmaxMatch = 1",
    "SpaceShower:pTmaxFudge = 1.",
    "SpaceShower:alphaSvalue = 0.118",
    "SpaceShower:alphaSorder = 1",
    "SpaceShower:alphaEMorder = 0",
    "SpaceShower:rapidityOrder = off",
    "SpaceShower:alphaSuseCMW = false",
    "TimeShower:MEcorrections = off",
    "SpaceShower:MEcorrections = off",
    "BeamRemnants:primordialKT = on",
    "PDF:pSet = LHAPDF6:NNPDF31_nlo_as_0118",
    "JetMatching:doFxFx = off",
)


def fxfx_available() -> bool:
    """Whether hep/ext/smlab_fxfx (PYTHIA's FxFx matching hook) is built."""
    import importlib.util

    return importlib.util.find_spec("smlab_fxfx") is not None


def nlo_samples(all_variants: bool = False) -> dict[str, dict]:
    """Available MC@NLO samples with their info.json.

    Keyed '<process>@<sqrt_s>' with the most accurate sample for that process;
    with ``all_variants``, keyed '<process>@<sqrt_s>#<sample name>' for every sample.
    """
    out: dict[str, dict] = {}
    if not os.path.isdir(NLO_DIR):
        return out
    entries = os.listdir(NLO_DIR)
    for process, names in NLO_SAMPLES.items():
        for name in reversed(names):  # later (better) names overwrite earlier ones
            prefix = name + "_"
            for entry in entries:
                energy = entry[len(prefix):]
                info_path = os.path.join(NLO_DIR, entry, "info.json")
                if entry.startswith(prefix) and energy.isdigit() and os.path.exists(info_path):
                    with open(info_path, encoding="utf-8") as handle:
                        info = json.load(handle)
                    if info.get("fxfx") and not fxfx_available():
                        continue  # showering it unmerged would double count jets
                    info["path"] = os.path.join(NLO_DIR, entry, "events.lhe")
                    info["sample"] = name
                    key = f"{process}@{int(energy)}"
                    out[f"{key}#{name}" if all_variants else key] = info
    return out


def smlab_tune() -> dict | None:
    """The Hadronica shower tune written by hep/tune.py (name, settings, applies_to), if any."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation", "tune.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        tune = json.load(handle)
    return {key: tune.get(key) for key in ("name", "settings", "applies_to", "data", "sample")}


def smlab_tune_settings(nlo: dict | None) -> dict:
    """Settings of the Hadronica tune for this MC@NLO sample; none for LO (Monash) runs."""
    tune = smlab_tune()
    if nlo is None or tune is None:
        return {}
    applies = tune.get("applies_to")
    if applies and nlo.get("sample") not in applies:
        return {}
    return dict(tune["settings"])


class Worker:
    def __init__(self) -> None:
        self.pythia = None
        self.config = None
        self.beam_kind = None
        self.jet_finder = None
        self.species_sent: set[int] = set()

    # -- helpers ------------------------------------------------------------

    def _info(self):
        pythia = self.pythia
        info = getattr(pythia, "info", None)
        if info is None or callable(info):
            info = pythia.infoPython()
        return info

    def hello(self) -> dict:
        probe = pythia8.Pythia(XMLDIR, False)
        version = probe.settings.parm("Pythia:versionNumber")
        return {
            "ok": True,
            "pythia": f"{version:.3f}",
            "processes": {key: {"title": p["title"], "beams": p["beams"]} for key, p in PROCESSES.items()},
            "nlo_samples": {key: {k: v for k, v in info.items() if k != "path"} for key, info in nlo_samples().items()},
            "tune": smlab_tune(),
            "python": sys.version.split()[0],
        }

    def init(self, config: dict) -> dict:
        process_id = config["process"]
        spec = PROCESSES[process_id]
        beams = config.get("beams", "ee")
        if spec["beams"] == "pp" and beams != "pp":
            raise ValueError(f"{process_id} needs proton beams")
        if spec["beams"] == "lepton" and beams not in LEPTON_BEAMS:
            raise ValueError(f"{process_id} needs lepton beams")
        pythia = pythia8.Pythia(XMLDIR, False)
        lines = ["Print:quiet = on", "Next:numberCount = 0"]
        nlo = None
        if config.get("source") == "nlo":
            key = f"{process_id}@{int(round(float(config['sqrt_s'])))}"
            if config.get("sample"):
                nlo = nlo_samples(all_variants=True).get(f"{key}#{config['sample']}")
            else:
                nlo = nlo_samples().get(key)
            if nlo is None:
                raise ValueError(f"no MC@NLO sample for {process_id} at {config['sqrt_s']} GeV")
        if nlo is not None:
            lines += list(MCATNLO_SETTINGS)
            lines += [f"Beams:LHEF = {nlo['path']}", f"Beams:nSkipLHEFatInit = {int(config.get('lhef_skip', 0))}"]
            if nlo.get("fxfx"):
                # FxFx merging exactly as MadGraph configures PYTHIA 8 (ickkw = 3).
                fxfx = nlo["fxfx"]
                lines += [
                    "JetMatching:doFxFx = on",
                    "JetMatching:merge = on",
                    f"JetMatching:qCut = {float(fxfx['qcut'])}",
                    f"JetMatching:qCutME = {float(fxfx['ptj'])}",
                    "JetMatching:coneRadius = 1.0",
                    "JetMatching:etaJetMax = 1000.0",
                    f"JetMatching:nJetMax = {int(fxfx['njmax'])}",
                    "JetMatching:scheme = 1",
                    "JetMatching:setMad = off",
                ]
                lines = [line for line in lines if line != "JetMatching:doFxFx = off"]
                # The JetMatching settings act only through PYTHIA's matching hook
                # (Pythia8Plugins/JetMatching.h), attached by hep/ext/smlab_fxfx.cpp.
                if not fxfx_available():
                    raise RuntimeError("FxFx sample needs PYTHIA's matching hook: run hep/ext/build_fxfx.sh")
                import smlab_fxfx

                smlab_fxfx.attach(pythia)
        elif beams == "pp":
            lines += ["Beams:idA = 2212", "Beams:idB = 2212"]
        else:
            id_a, id_b = LEPTON_BEAMS[beams]
            lines += [f"Beams:idA = {id_a}", f"Beams:idB = {id_b}"]
            if not config.get("isr", True):
                lines += ["PDF:lepton = off"]
        if nlo is None:
            lines.append(f"Beams:eCM = {float(config['sqrt_s']):.6f}")
        lines += list(PDG_OVERRIDES)
        # Decay a particle only if its sampled decay point lies inside the tracker
        # cylinder (R = 1.2 m, |z| = 1.2 m · sinh 1.5), so K0_S, Λ, and b/c-hadron
        # decays appear as real displaced vertices and longer-lived particles
        # (π±, K±, K0_L, n, μ) reach the calorimeters and muon system.
        # With decays = "generator" (used for Rivet validation), PYTHIA's standard
        # decay table applies unchanged: K0_S, Λ, and other hyperons decay, and
        # π±, K±, K0_L, and n are stable, the convention of particle-level measurements.
        if config.get("decays", "detector") == "detector":
            lines += [
                "ParticleDecays:limitCylinder = on",
                "ParticleDecays:xyMax = 1200.",
                "ParticleDecays:zMax = 2556.",
            ]
        if config.get("pdf") and beams == "pp" and nlo is None:
            lines.append(f"PDF:pSet = LHAPDF6:{config['pdf']}")
        if not config.get("hadronize", True):
            lines.append("HadronLevel:all = off")
        if not config.get("mpi", True) and beams == "pp":
            lines.append("PartonLevel:MPI = off")
        if nlo is None:
            lines += spec["settings"]
        else:
            # The hard process comes from the file; keep only the decay-mode choices.
            lines += [line for line in spec["settings"] if line.split(":")[0].strip().isdigit()]
        # Shower-tune overrides (see hep/tune.py), applied last. "smlab" selects the
        # tune in validation/tune.json, which was fitted on top of the MC@NLO settings.
        tune = config.get("tune") or {}
        if tune == "smlab":
            tune = smlab_tune_settings(nlo)
        lines += [f"{key} = {value}" for key, value in tune.items()]
        lines += ["Random:setSeed = on", f"Random:seed = {int(config.get('seed', 1)) % 900000000}"]
        for line in lines:
            if not pythia.readString(line):
                raise ValueError(f"PYTHIA rejected setting: {line}")
        if not pythia.init():
            raise RuntimeError("PYTHIA initialization failed; see the log")
        self.pythia = pythia
        self.species_sent = set()
        self.config = dict(config)
        self.nlo = nlo
        self.beam_kind = beams
        if beams == "pp":
            # FastJet anti-kT, R = 0.4, pT > 20 GeV, |η| < 4.7: the standard LHC jet definition.
            self.jet_finder = ("antikt", fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4))
        else:
            # FastJet e+e- k_T (Durham), exclusive jets at y_cut = 0.005.
            self.jet_finder = ("durham", fastjet.JetDefinition(fastjet.ee_kt_algorithm))
        return {"ok": True, "settings": lines, "beams": beams,
                "nlo": {k: v for k, v in nlo.items() if k != "path"} if nlo else None}

    def generate(self, n: int, jets: bool = True) -> dict:
        if self.pythia is None:
            raise RuntimeError("init first")
        events = []
        started = time.perf_counter()
        failures = 0  # consecutive
        vetoed = 0  # total, for FxFx bookkeeping
        while len(events) < n:
            if not self.pythia.next():
                if self.nlo is not None and self._info().atEndOfFile():
                    # A finite MC@NLO sample: start it again with a new shower seed.
                    self.rewinds = getattr(self, "rewinds", 0) + 1
                    config = {**self.config, "seed": int(self.config.get("seed", 1)) + 7919 * self.rewinds,
                              "lhef_skip": 0}
                    rewinds = self.rewinds
                    self.init(config)
                    self.rewinds = rewinds
                    continue
                failures += 1
                vetoed += 1
                # FxFx vetoes a fraction of the showered events by design.
                limit = 5000 if self.nlo is not None and self.nlo.get("fxfx") else 20
                if failures > limit:
                    raise RuntimeError(f"PYTHIA failed to generate an event {limit} times in a row")
                continue
            failures = 0
            events.append(self._serialize(jets))
        detector = None
        if self.config.get("detector"):
            # Delphes fast simulation with the card that matches this collider.
            from detector import simulate

            recos, detector = simulate(
                events, self.beam_kind, float(self.config["sqrt_s"]), float(self.config.get("pileup", 0.0))
            )
            for event, reco in zip(events, recos):
                event["reco"] = reco
        info = self._info()
        return {
            "vetoed": vetoed,
            "detector": detector,
            "nlo": {k: v for k, v in self.nlo.items() if k != "path"} if self.nlo else None,
            "ok": True,
            "events": events,
            "sigma_pb": info.sigmaGen() * MB_TO_PB,
            "sigma_err_pb": info.sigmaErr() * MB_TO_PB,
            "n_accepted": info.nAccepted(),
            "seconds": time.perf_counter() - started,
        }

    def _jets(self, event) -> list[list[float]]:
        """Cluster visible final-state particles (neutrinos excluded)."""
        kind, definition = self.jet_finder
        inputs = []
        for i in range(event.size()):
            p = event[i]
            if not p.isFinal() or not p.isVisible():
                continue
            if kind == "antikt" and abs(p.eta()) > 4.7:
                continue
            inputs.append(fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e()))
        if len(inputs) < 2:
            return []
        sequence = fastjet.ClusterSequence(inputs, definition)
        if kind == "antikt":
            found = fastjet.sorted_by_pt(sequence.inclusive_jets(20.0))
            found = [j for j in found if abs(j.eta()) < 4.7]
        else:
            found = sequence.exclusive_jets_ycut(0.005)
        return [
            [round(j.px(), 5), round(j.py(), 5), round(j.pz(), 5), round(j.E(), 5), len(j.constituents())]
            for j in found
        ]

    def _serialize(self, with_jets: bool = True) -> dict:
        event = self.pythia.event
        process = self.pythia.process
        particles = []
        species = {}
        data = self.pythia.particleData
        for i in range(event.size()):
            p = event[i]
            pid = p.id()
            if pid not in self.species_sent and pid != 90:
                self.species_sent.add(pid)
                # Name, nominal mass (GeV), and proper cτ (mm) from PYTHIA's particle table.
                species[pid] = [data.name(pid), data.m0(pid), data.tau0(pid)]
            particles.append([
                p.id(), p.status(), p.mother1(), p.mother2(), p.daughter1(), p.daughter2(),
                round(p.px(), 7), round(p.py(), 7), round(p.pz(), 7), round(p.e(), 7), round(p.m(), 7),
                round(p.xProd(), 6), round(p.yProd(), 6), round(p.zProd(), 6), round(p.tau(), 6),
                round(p.charge(), 4),
            ])
        hard = []
        for i in range(process.size()):
            p = process[i]
            hard.append([p.id(), p.status(), p.mother1(), p.mother2(),
                         round(p.px(), 6), round(p.py(), 6), round(p.pz(), 6), round(p.e(), 6), round(p.m(), 6)])
        info = self._info()
        jets = self._jets(event) if with_jets else []
        return {
            "species": species,
            "particles": particles,
            "hard": hard,
            "jets": jets,
            "code": info.code(),
            "name": info.name(),
            "sqrt_s_hat": math.sqrt(max(info.sHat(), 0.0)) if hasattr(info, "sHat") else None,
            "weight": info.weight(),
        }


def main() -> None:
    # PYTHIA's C++ code prints to file descriptor 1. Keep a private copy of
    # stdout for the JSON protocol and send everything else to stderr.
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    worker = Worker()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request: dict = {}
        try:
            request = json.loads(raw)
            cmd = request.get("cmd")
            if cmd == "hello":
                reply = worker.hello()
            elif cmd == "init":
                reply = worker.init(request["config"])
            elif cmd == "generate":
                reply = worker.generate(int(request.get("n", 1)))
            elif cmd == "quit":
                out.write(json.dumps({"ok": True, "bye": True}) + "\n")
                out.flush()
                return
            else:
                reply = {"ok": False, "error": f"unknown command {cmd!r}"}
        except Exception as exc:  # the UI shows the message; keep serving
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        if isinstance(request, dict) and "id" in request:
            reply["id"] = request["id"]
        out.write(json.dumps(reply, separators=(",", ":")) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
